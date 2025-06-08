import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === Load trained YOLO model ===
model = YOLO('PIPE.pt')

# === Initialize webcam ===
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Optional: Save output ===
save_output = False
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('webcam_yolo_output.mp4', fourcc, 20.0, (640, 480))

# === Create matplotlib figure ===
plt.ion()  # Turn interactive mode on
fig, ax = plt.subplots()

im = None  # Placeholder for the image object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False, show=False)[0]

    for i, box in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, conf, cls = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Draw box, center and ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        # Get class name from ID
        class_id = int(cls)
        class_name = model.names[class_id] if hasattr(model, 'names') else str(class_id)
        label = f"{class_name} ({conf:.2f})"

        # Draw label
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


    # Convert BGR (OpenCV) to RGB (Matplotlib)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if im is None:
        im = ax.imshow(frame_rgb)
    else:
        im.set_data(frame_rgb)

    plt.pause(0.001)  # Required for real-time update

    if save_output:
        out.write(frame)

    # Break on Ctrl+C in terminal or manual stop
    if plt.get_fignums() == []:
        break

# === Cleanup ===
cap.release()
if save_output:
    out.release()
plt.ioff()
plt.close()
