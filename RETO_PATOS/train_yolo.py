from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # or yolov8s.yaml for more capacity
model.train(data="RETO_PATOS/YOLO_dataset_improved_separated/data.yaml", epochs=50, imgsz=640)
