from ultralytics import YOLO, RTDETR
import os

DATA = r"D:\Training + AI\PPE Train\datasets\PPE-3\data.yaml"

MODEL_PATHS = {
    "YOLOv8": r"D:\Training + AI\PPE Train\Benchmark\YOLOv8\yolov8_best.pt",
    "YOLO26": r"D:\Training + AI\PPE Train\Benchmark\YOLO26\yolo26_best.pt",
    "RT-DETR": r"D:\Training + AI\PPE Train\Benchmark\RT-DETR\rtdetr_best.pt",
}


def evaluate_model(name, path):
    print(f"\n{'='*50}")
    print(name)
    print('='*50)

    if name == "RT-DETR":
        model = RTDETR(path)
    else:
        model = YOLO(path)

    metrics = model.val(
        data=DATA,
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        verbose=True
    )

    P = metrics.box.mp
    R = metrics.box.mr
    F1 = 2 * P * R / (P + R + 1e-9)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    print("\nRESULT SUMMARY")
    print(f"Model      : {name}")
    print(f"Precision  : {P:.4f}")
    print(f"Recall     : {R:.4f}")
    print(f"mAP50      : {metrics.box.map50:.4f}")
    print(f"mAP50-95   : {metrics.box.map:.4f}")
    print(f"F1-score   : {F1:.4f}")
    print(f"Params     : {params:.2f} M")
    print(f"Model Size : {size_mb:.2f} MB")


if __name__ == "__main__":
    for name, path in MODEL_PATHS.items():
        evaluate_model(name, path)