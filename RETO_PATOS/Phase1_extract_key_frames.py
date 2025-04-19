import cv2
import os

video_path = 'RETO_PATOS/patos_sin_sal.mp4'
save_dir = 'RETO_PATOS/frames_key'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_rate = 10  # save every n frames
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if i % frame_rate == 0:
        cv2.imwrite(f"{save_dir}/frame_{i:04d}.jpg", frame)
    i += 1

cap.release()
