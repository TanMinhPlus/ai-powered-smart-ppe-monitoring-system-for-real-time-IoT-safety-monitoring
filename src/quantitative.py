from ultralytics import YOLO, RTDETR
import os
import time
import numpy as np

DATA = r"D:\Training + AI\PPE Train\datasets\PPE-3\data.yaml"

MODEL_PATHS = {
    "YOLOv8": r"D:\Training + AI\PPE Train\Benchmark\YOLOv8\yolov8_best.pt",
    "YOLO26": r"D:\Training + AI\PPE Train\Benchmark\YOLO26\yolo26_best.pt",
    "RT-DETR": r"D:\Training + AI\PPE Train\Benchmark\RT-DETR\rtdetr_best.pt",
}


def load_model(name, path):
    return RTDETR(path) if name == "RT-DETR" else YOLO(path)


def benchmark_fps(model, n=200):
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    for _ in range(20):
        model.predict(img, imgsz=640, device=0, verbose=False)

    start = time.time()

    for _ in range(n):
        model.predict(img, imgsz=640, device=0, verbose=False)

    elapsed = time.time() - start
    return n / elapsed


def evaluate_model(name, path):
    print(f"\n{'='*50}")
    print(name)
    print('='*50)

    model = load_model(name, path)

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

    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    fps = benchmark_fps(model)

    print("\nTABLE 4 VALUES")
    print(f"Model/Method : {name}")
    print(f"Main Metric  : {metrics.box.map:.4f}")
    print(f"Precision    : {P:.4f}")
    print(f"Recall       : {R:.4f}")
    print(f"F1-score     : {F1:.4f}")
    print(f"Params       : {params:.2f} M")
    print(f"FPS          : {fps:.2f}")


if __name__ == "__main__":
    for name, path in MODEL_PATHS.items():
        evaluate_model(name, path)