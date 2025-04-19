from ultralytics import YOLO
import cv2
import subprocess

# 1. Cargar modelo
model = YOLO("yolov8n.pt")

# 2. Abrir video de entrada
video_path = "RETO_PATOS/patos_sin_sal.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Crear video de salida
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("RETO_PATOS/patos_detectados.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 4. Ejecutar YOLO sobre el frame
    results = model.predict(source=frame, verbose=False)
    annotated_frame = results[0].plot()

    # 5. Guardar en video de salida
    out.write(annotated_frame)
    frame_count += 1

    # (opcional) print progreso
    if frame_count % 10 == 0:
        print(f"🎥 Procesado frame {frame_count}")

cap.release()
out.release()
print("✅ Video con detecciones guardado en: RETO_PATOS/patos_detectados.mp4")

# xdg-open RETO_PATOS/patos_detectados.mp4

# Ruta al video
video_path = "RETO_PATOS/patos_detectados.mp4"

# Ejecutar el comando en terminal para abrir el video con el visor predeterminado
subprocess.run(["xdg-open", video_path])