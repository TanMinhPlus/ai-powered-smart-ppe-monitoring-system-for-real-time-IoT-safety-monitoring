import json
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\OneDrive\Document\PPE Train\datasets\PPE_COCO_DATA")

for split in ["train", "valid", "test"]:
    path = ROOT / split / "_annotations_fixed.coco.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_map = {img["id"]: img for img in data["images"]}

    clean_anns = []
    removed = 0

    for ann in data["annotations"]:
        img = image_map.get(ann["image_id"])
        if img is None:
            removed += 1
            continue

        x, y, w, h = ann["bbox"]
        W, H = img["width"], img["height"]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)

        new_w = x2 - x1
        new_h = y2 - y1

        if new_w <= 1 or new_h <= 1:
            removed += 1
            continue

        if ann["category_id"] < 0 or ann["category_id"] > 6:
            removed += 1
            continue

        ann["bbox"] = [x1, y1, new_w, new_h]
        ann["area"] = new_w * new_h
        clean_anns.append(ann)

    data["annotations"] = clean_anns

    out = ROOT / split / "_annotations_clean.coco.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(split, "removed:", removed, "kept:", len(clean_anns), "saved:", out)