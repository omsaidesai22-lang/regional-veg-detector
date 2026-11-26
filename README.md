# Regional Vegetable Detection & Stock Counter (YOLOv8s + FastAPI)

A custom end-to-end Object Detection web application built with **YOLOv8s**, **FastAPI**, and **HTML/CSS/JavaScript** to detect, classify, and count 5 regional vegetables in real time and generate an inventory stock summary.

## Custom Target Classes (Non-COCO)
1. **Bitter Gourd (`bitter_gourd`)**
2. **Okra (`okra`)**
3. **Brinjal (`brinjal`)**
4. **Ridge Gourd (`ridge_gourd`)**
5. **Bottle Gourd (`bottle_gourd`)**

## Project Pipeline
1. **Data Collection (`1_scrape_veggies.py`)**: Uses `icrawler` (`BingImageCrawler`) to collect ~95 images per vegetable class.
2. **Auto-Labeling & Multi-Count Augmentation (`2_prepare_dataset.py`)**: Extracts clean foreground bounding boxes, generates synthetic multi-vegetable scenes for multi-object counting, splits data 80/20 (`train`/`val`), and exports `yolo_dataset.zip`.
3. **Model Training (Google Colab)**: Fine-tunes `yolov8s.pt` at `imgsz=640` for `40` epochs on a T4 GPU.
4. **Web Application (`app.py` & `index.html`)**: FastAPI inference server running with a `0.65` confidence threshold, paired with a dark-mode drag-and-drop dashboard displaying bounding boxes, match confidence, item counts, and a stock summary table.

## Model Evaluation Results
* **Precision:** `0.XX`
* **Recall:** `0.XX`
* **mAP@50:** `0.XX`
* **mAP@50-95:** `0.XX`

## Local Setup & Execution
```bash
pip install fastapi uvicorn ultralytics opencv-python-headless python-multipart
python app.py