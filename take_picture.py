import cv2
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

save_dir = "data/data6"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("📸 Press 's' to save an image | Press 'q' to quit")
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    cv2.imshow("Camera Preview", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        filename = os.path.join(save_dir, f"capture_{i:03d}.png")
        cv2.imwrite(filename, frame)
        print(f"✅ Saved {filename}")
        i += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
