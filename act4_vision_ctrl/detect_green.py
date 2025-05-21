import matplotlib.pyplot as plt
import matplotlib.animation as animation
from djitellopy import Tello
import cv2
import numpy as np

# Connect to Tello and start stream (no takeoff)
tello = Tello(host="192.168.12.6")
tello.connect()
tello.streamon()

# Initial frame for shape and display setup
frame = tello.get_frame_read().frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots()
im = ax.imshow(frame_rgb)
ax.axis("off")

# Define HSV bounds for light green (adjust manually)
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Store latest frame globally for mouse callback
latest_frame = frame_rgb.copy()

# Mouse event to show RGB values in window title
def on_mouse_move(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < latest_frame.shape[0] and 0 <= x < latest_frame.shape[1]:
            r, g, b = latest_frame[y, x]
            fig.canvas.manager.set_window_title(f"RGB at ({x},{y}): R={r}, G={g}, B={b}")

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Frame update function
def update(*args):
    global latest_frame
    frame = tello.get_frame_read().frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Detect green centroid
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    latest_frame = frame.copy()  # for RGB probe (still in BGR)
    im.set_array(frame)
    return [im]


# Start live animation
ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

try:
    plt.show()
finally:
    tello.streamoff()
    tello.end()
