from ultralytics import YOLO, RTDETR

TEST_IMAGES = r"D:\Training + AI\PPE Train\datasets\PPE-3\test\images"

models = {
    "yolov8": YOLO(r"D:\Training + AI\PPE Train\Benchmark\YOLOv8\yolov8_best.pt"),
    "yolo26": YOLO(r"D:\Training + AI\PPE Train\Benchmark\YOLO26\yolo26_best.pt"),
    "rtdetr": RTDETR(r"D:\Training + AI\PPE Train\Benchmark\RT-DETR\rtdetr_best.pt"),
}

for name, model in models.items():
    model.predict(
        source=TEST_IMAGES,
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        project=r"D:\Training + AI\PPE Train\qualitative_results",
        name=name
    )