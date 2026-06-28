from ultralytics import YOLO, RTDETR
import os

MODELS = {
    "YOLOv8": r"D:\Training + AI\PPE Train\Benchmark\YOLOv8\yolov8_best.pt",
    "YOLO26": r"D:\Training + AI\PPE Train\Benchmark\YOLO26\yolo26_best.pt",
    "RT-DETR": r"D:\Training + AI\PPE Train\Benchmark\RT-DETR\rtdetr_best.pt",
}

for name, path in MODELS.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model = RTDETR(path) if name == "RT-DETR" else YOLO(path)

    # Parameters
    params = sum(p.numel() for p in model.model.parameters()) / 1e6

    # Model Size
    model_size = os.path.getsize(path) / (1024 * 1024)

    print(f"Parameters : {params:.2f} M")
    print(f"Model Size : {model_size:.2f} MB")

    print("\nModel Summary:")
    model.info()