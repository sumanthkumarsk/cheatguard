import React, { useRef, useEffect, useState } from "react";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [detections, setDetections] = useState([]);
  const [flagged, setFlagged] = useState(false);
  const BACKEND = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

  useEffect(() => {
    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      } catch (e) {
        console.error("Could not start video", e);
      }
    }
    start();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      captureAndSend();
    }, 1000); // send 1 frame per second
    return () => clearInterval(interval);
  }, []);

  async function captureAndSend() {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.85));
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.85));

    const form = new FormData();
    form.append("file", blob, "frame.jpg");

    try {
      const r = await fetch(`${BACKEND}/detect`, { method: "POST", body: form });
      if (!r.ok) throw new Error("Backend error");
      const data = await r.json();
      setDetections(data.detections || []);
      setFlagged(data.flagged || false);
      drawBoxes(data.detections || []);
    } catch (e) {
      console.error("Detection request failed:", e);
    }
  }

  function drawBoxes(dets) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.font = "16px Arial";
    dets.forEach(d => {
      const [x1, y1, x2, y2] = d.bbox;
      ctx.strokeStyle = "red";
      ctx.fillStyle = "red";
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillText(`${d.class} ${(d.confidence * 100).toFixed(1)}%`, x1 + 4, y1 + 16);
    });
  }

  return (
    <div style={{ padding: 12 }}>
      <h2>Live Detection Frontend</h2>
      <div style={{ position: "relative", width: 640 }}>
        <video ref={videoRef} style={{ width: 640 }} />
        <canvas ref={canvasRef} style={{ position: "absolute", left: 0, top: 0 }} />
      </div>
      <div style={{ marginTop: 10 }}>
        <strong>Flagged:</strong> {flagged ? "YES" : "no"}
      </div>
      <pre style={{ maxHeight: 200, overflow: "auto" }}>{JSON.stringify(detections, null, 2)}</pre>
    </div>
  );
}

export default App;
