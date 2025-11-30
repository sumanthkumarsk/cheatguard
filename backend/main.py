import os
import io
import time
import datetime
import json
from typing import List, Dict, Any

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

LOG_PATH = os.environ.get("LOG_PATH", "logs/events.log")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
# Comma-separated class names to flag (e.g. "cell phone,person")
FLAG_CLASSES = [s.strip().lower() for s in os.environ.get("FLAG_CLASSES", "cell phone").split(",") if s.strip()]

os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

app = FastAPI(title="Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once on startup
try:
    model = YOLO(MODEL_PATH)
    names = {i: n.lower() for i, n in model.model.names.items()} if hasattr(model, "model") and hasattr(model.model, "names") else model.names
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

def write_log(entry: Dict[str, Any]):
    line = json.dumps(entry, default=str)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

@app.get("/")
async def root():
    return {"status": "ok", "model": MODEL_PATH, "flag_classes": FLAG_CLASSES}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # read bytes
    b = await file.read()
    if not b:
        raise HTTPException(status_code=400, detail="Empty file")
    # decode to numpy image
    npimg = np.frombuffer(b, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode image")
    start = time.time()
    # model inference
    results = model(frame, imgsz=1024)  # adjust imgsz if needed
    infer_time = (time.time() - start)
    detections = []
    flagged = False
    flagged_reasons = []
    for r in results:  # usually results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        # boxes.xyxy, boxes.conf, boxes.cls
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.array([])
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.array([])
        clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.array([])
        for box, conf, cls in zip(xyxy, confs, clss):
            cls_name = names.get(int(cls), str(cls)).lower()
            x1, y1, x2, y2 = [float(x) for x in box]
            det = {
                "class": cls_name,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2]
            }
            detections.append(det)
            # check flag rule
            if FLAG_CLASSES and any(fc in cls_name for fc in FLAG_CLASSES):
                flagged = True
                flagged_reasons.append(cls_name)
    response = {
        "detections": detections,
        "inference_seconds": infer_time,
        "flagged": flagged,
        "flagged_reasons": flagged_reasons
    }
    if flagged:
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "flagged_reasons": flagged_reasons,
            "detections": detections,
            "note": "Rule-based flag triggered"
        }
        write_log(log_entry)
    return response

@app.get("/logs")
async def get_logs(limit: int = 100):
    # return last N lines
    if not os.path.exists(LOG_PATH):
        return {"logs": []}
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()[-limit:]
    return {"logs": [json.loads(l) for l in lines]}
