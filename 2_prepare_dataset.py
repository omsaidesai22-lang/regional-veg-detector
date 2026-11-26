import os
import glob
import random
import shutil
import zipfile
import cv2
import numpy as np

RAW_DIR = "raw_dataset"
DATASET_DIR = "yolo_dataset"
ZIP_NAME = "yolo_dataset.zip"

CLASSES = ["bitter_gourd", "okra", "brinjal", "ridge_gourd", "bottle_gourd"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}

def get_foreground_bbox(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(blurred, 235, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)
    area_ratio = (bw * bh) / float(w * h)

    if not (0.15 <= area_ratio <= 0.88):
        return None

    x_center = (x + bw / 2.0) / w
    y_center = (y + bh / 2.0) / h
    norm_w = bw / float(w)
    norm_h = bh / float(h)
    return (x_center, y_center, norm_w, norm_h), (x, y, bw, bh)

def create_multi_object_collages(single_items, num_collages=160):
    collages = []
    canvas_size = 640

    for i in range(num_collages):
        bg_color = random.randint(190, 240)
        canvas = np.full((canvas_size, canvas_size, 3), bg_color, dtype=np.uint8)
        noise = np.random.randint(-8, 8, (canvas_size, canvas_size, 3), dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        num_objs = random.randint(2, 6)
        placed_boxes = []
        labels = []

        for _ in range(num_objs):
            cls_id, crop = random.choice(single_items)
            scale = random.uniform(0.25, 0.42)
            ch, cw = crop.shape[:2]
            aspect = cw / float(ch)
            new_h = int(canvas_size * scale)
            new_w = int(new_h * aspect)
            if new_w >= canvas_size - 20 or new_h >= canvas_size - 20:
                continue

            resized = cv2.resize(crop, (new_w, new_h))

            for _attempt in range(15):
                px = random.randint(10, canvas_size - new_w - 10)
                py = random.randint(10, canvas_size - new_h - 10)
                overlap = False
                for (bx, by, bw, bh) in placed_boxes:
                    if not (px + new_w < bx or px > bx + bw or py + new_h < by or py > by + bh):
                        overlap = True
                        break
                if not overlap:
                    mask = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) < 235
                    roi = canvas[py:py+new_h, px:px+new_w]
                    roi[mask] = resized[mask]
                    canvas[py:py+new_h, px:px+new_w] = roi

                    placed_boxes.append((px, py, new_w, new_h))
                    xc = (px + new_w / 2.0) / canvas_size
                    yc = (py + new_h / 2.0) / canvas_size
                    nw = new_w / float(canvas_size)
                    nh = new_h / float(canvas_size)
                    labels.append((cls_id, xc, yc, nw, nh))
                    break

        if len(labels) >= 2:
            collages.append((f"collage_{i}.jpg", canvas, labels))
    return collages

def build_dataset():
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    all_samples = []
    single_crops = []

    for cls_name in CLASSES:
        cls_id = CLASS_TO_ID[cls_name]
        folder = os.path.join(RAW_DIR, cls_name)
        image_paths = glob.glob(os.path.join(folder, "*.*"))
        kept = 0

        for idx, path in enumerate(image_paths):
            img = cv2.imread(path)
            if img is None:
                continue
            res = get_foreground_bbox(img)
            if res is None:
                continue

            yolo_box, (x, y, bw, bh) = res
            fname = f"{cls_name}_{idx}.jpg"
            all_samples.append((fname, img, [(cls_id, *yolo_box)]))
            single_crops.append((cls_id, img[y:y+bh, x:x+bw]))
            kept += 1

        print(f"[{cls_name}] Filtered & labeled: {kept} images")

    if single_crops:
        collages = create_multi_object_collages(single_crops, num_collages=160)
        all_samples.extend(collages)
        print(f"[Multi-Count Augmentation] Added {len(collages)} multi-vegetable images")

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.8)
    splits = {"train": all_samples[:split_idx], "val": all_samples[split_idx:]}

    for split_name, items in splits.items():
        for fname, img, boxes in items:
            img_out = os.path.join(DATASET_DIR, "images", split_name, fname)
            lbl_out = os.path.join(DATASET_DIR, "labels", split_name, os.path.splitext(fname)[0] + ".txt")
            cv2.imwrite(img_out, img)
            with open(lbl_out, "w") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

    yaml_content = f"""path: /content/yolo_dataset
train: images/train
val: images/val
nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(DATASET_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ".")
                zipf.write(full_path, rel_path)

    print(f"\nSuccess! Created {ZIP_NAME} ({len(splits['train'])} train, {len(splits['val'])} val).")

if __name__ == "__main__":
    build_dataset()