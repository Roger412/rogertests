from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # or yolov8s.yaml for more capacity
model.train(data="RETO_PATOS/patos_dataset/data.yaml", epochs=50, imgsz=640)
