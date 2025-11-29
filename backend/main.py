from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import time
import os

app = FastAPI()

model = YOLO("models/best.pt")  # your cheating model

LOG_FILE = "cheating_log.txt"
os.makedirs("logs", exist_ok=True)
log_path = "logs/cheating_log.txt"

def write_log(message: str):
    with open(log_path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()

    np_img = cv2.imdecode(
        np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR
    )

    results = model(np_img)

    cheating_detected = False

    for r in results:
        for c in r.boxes.cls:
            cheating_detected = True
            write_log("Cheating detected")

    return {"cheating": cheating_detected}
