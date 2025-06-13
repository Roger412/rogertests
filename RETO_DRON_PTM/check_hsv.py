import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Initialize webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# globals to hold the latest frames
latest_bgr = None
latest_hsv = None

# === Matplotlib setup ===
fig, ax = plt.subplots()
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot read initial frame from webcam")

latest_bgr = frame.copy()
latest_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

im = ax.imshow(frame_rgb)
ax.axis('off')
fig.canvas.manager.set_window_title("HSV Viewer")

# === Update function ===
def update(_):
    global latest_bgr, latest_hsv
    ret, frame = cap.read()
    if not ret:
        return [im]
    latest_bgr = frame.copy()
    latest_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return [im]

# === Mouse move handler ===
def on_mouse_move(event):
    if event.inaxes is not ax or latest_hsv is None:
        return
    x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
    h, s, v = latest_hsv[y, x]
    fig.canvas.manager.set_window_title(f"HSV at ({x},{y}): H={h}, S={s}, V={v}")

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# === Run the animation ===
ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()

# === Cleanup ===
cap.release()
