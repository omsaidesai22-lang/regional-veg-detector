import os
import glob
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

app = FastAPI(title="Regional Vegetable Detection & Counting API")

weight_files = sorted(glob.glob("best*.pt"))
if not weight_files:
    raise FileNotFoundError("No 'best*.pt' model file found in the current directory!")

MODEL_PATH = weight_files[0]
print(f"Loading custom YOLOv8s weights from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

DISPLAY_NAMES = {
    "bitter_gourd": ("Gourd Family", "Bitter Gourd (Karela)"),
    "okra": ("Pod Vegetable", "Okra (Bhindi)"),
    "brinjal": ("Nightshade", "Brinjal (Eggplant)"),
    "ridge_gourd": ("Gourd Family", "Ridge Gourd (Turai)"),
    "bottle_gourd": ("Gourd Family", "Bottle Gourd (Lauki)")
}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/detect")
async def detect_vegetables(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.")

    results = model.predict(source=img, conf=0.65, iou=0.45, verbose=False)
    result = results[0]

    annotated_img = result.plot(line_width=2, font_size=1.0)
    _, buffer = cv2.imencode(".jpg", annotated_img)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    detections = []
    stock_counts = {}

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        raw_label = model.names[cls_id]
        confidence = round(float(box.conf[0].item()) * 100, 1)

        category, clean_name = DISPLAY_NAMES.get(
            raw_label, ("Regional Produce", raw_label.replace("_", " ").title())
        )

        detections.append({
            "category": category,
            "name": clean_name,
            "raw_label": raw_label,
            "confidence": confidence
        })

        stock_counts[clean_name] = stock_counts.get(clean_name, 0) + 1

    detections.sort(key=lambda x: x["confidence"], reverse=True)

    return JSONResponse({
        "annotated_image": f"data:image/jpeg;base64,{encoded_image}",
        "total_count": len(detections),
        "stock_summary": stock_counts,
        "detections": detections
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)