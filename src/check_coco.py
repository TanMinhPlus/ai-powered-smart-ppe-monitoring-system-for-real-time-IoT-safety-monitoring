# check_coco.py

import json

ann = r"C:\Users\ADMIN\OneDrive\Document\PPE Train\datasets\PPE_COCO_DATA\train\_annotations.coco.json"

with open(ann, "r", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 50)
print("CATEGORIES")
print("=" * 50)

for c in data["categories"]:
    print(c)

print()

cat_ids = sorted([c["id"] for c in data["categories"]])
ann_ids = sorted(set([a["category_id"] for a in data["annotations"]]))

print("Category IDs :", cat_ids)
print("Annotation IDs:", ann_ids)

print("Num Categories:", len(cat_ids))
print("Num Images    :", len(data["images"]))
print("Num Annots    :", len(data["annotations"]))