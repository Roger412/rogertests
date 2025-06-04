import matplotlib.pyplot as plt
import matplotlib.animation as animation
from djitellopy import Tello
import cv2
import numpy as np

# === Green circle detector with distance estimation ===
def detect_green_circles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=50, minRadius=40, maxRadius=150)
    diameters = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if mask[y, x] > 0:
                diameter = 2 * r
                distance_cm = (18 * 200) / diameter
                diameters.append((diameter, distance_cm, (x, y, r)))

        diameters.sort(reverse=True)

        for diameter, distance_cm, (x, y, r) in diameters:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            text = f"{diameter}px | {distance_cm:.1f}cm"
            cv2.putText(image, text, (x - r, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image

# === Tello setup ===
tello = Tello() #host="192.168.12.6"
tello.connect()
tello.streamon()
tello.set_video_direction(Tello.CAMERA_FORWARD)

# === Matplotlib setup ===
fig, ax = plt.subplots()
frame = tello.get_frame_read().frame
im = ax.imshow(frame)
ax.axis("off")

def update(*args):
    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))

    # Draw circles
    frame = detect_green_circles(frame)

    # Convert for matplotlib display
    im.set_array(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

try:
    plt.show()
finally:
    tello.streamoff()
    tello.end()
