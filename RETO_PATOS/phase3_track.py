import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Cargar YOLOv8 (usa el modelo 'yolov8n.pt' o 'yolov8s.pt')
model = YOLO('yolov8s.pt')  # Cambia si quieres otro

# Iniciar DeepSORT
tracker = DeepSort(max_age=30)

# Cargar homografía
H = np.load("RETO_PATOS/homography_matrix/homography_matrix.npy")

def pixel_to_world(pt, H):
    pt_h = np.array([pt[0], pt[1], 1.0])
    world = H @ pt_h
    world /= world[2]
    return world[:2]

# Abrir video
cap = cv2.VideoCapture("RETO_PATOS/patos_sin_sal.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Paso 1: YOLOv8 detección
    results = model(frame)[0]
    
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result
        class_name = model.names[int(cls)]
        if class_name in ["bird", "duck"]:  # ajusta si no detecta bien
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

    # Paso 2: DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Transformar centroide a coordenadas reales
        real_coords = pixel_to_world((cx, cy), H)
        print(f"🟡 Duck {track_id} at: {real_coords}")

        # Dibujar
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("Tracking patos con YOLOv8 + DeepSORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
                