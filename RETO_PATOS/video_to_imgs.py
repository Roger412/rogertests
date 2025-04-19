import cv2
import os

def extract_frames(video_path, output_folder, prefix="frame"):
    # Crear la carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Cargar el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ No se pudo abrir el video: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin del video

        # Crear nombre de archivo
        filename = os.path.join(output_folder, f"{prefix}_{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Guardado: {filename}")
        frame_count += 1

    cap.release()
    print(f"\n🎉 Se extrajeron {frame_count} frames en: {output_folder}")

# 🧪 Ejemplo de uso:
video_path = "RETO_PATOS/patos_sin_sal.mp4"
output_folder = "RETO_PATOS/video_imgs"
extract_frames(video_path, output_folder)
