from djitellopy import Tello
from pynput import keyboard
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import numpy as np

# Initialize drone
tello = Tello(host="192.168.12.6")
tello.connect()
tello.streamon()
tello.takeoff()

# Default camera direction
current_camera = Tello.CAMERA_FORWARD
tello.set_video_direction(current_camera)
frame_read = tello.get_frame_read()

# Velocity variables
forward_backward = 0
forward_backward_magnitude = 0
left_right = 0
left_right_magnitude = 0
up_down = 0
up_down_magnitude = 0
yaw = 0
yaw_magnitude = 0

lock = threading.Lock()

# HSV bounds for green
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# RC control loop
def control_loop():
    while True:
        with lock:
            tello.send_rc_control(int(left_right), int(forward_backward), int(up_down), int(yaw))
        time.sleep(0.05)

threading.Thread(target=control_loop, daemon=True).start()

# Key press handler
def on_press(key):
    global left_right, up_down, yaw, forward_backward
    global forward_backward_magnitude, left_right_magnitude, up_down_magnitude, yaw_magnitude
    global current_camera

    try:
        k = key.char.lower()
        with lock:
            if k == 'w':
                forward_backward = forward_backward_magnitude
            elif k == 's':
                forward_backward = -forward_backward_magnitude
            elif k == 'r':
                if forward_backward_magnitude + 2 <= 100:
                    forward_backward_magnitude += 2
            elif k == 'f':
                if forward_backward_magnitude - 2 >= 0:
                    forward_backward_magnitude -= 2
            elif k == 'e':
                yaw = yaw_magnitude
            elif k == 'q':
                yaw = -yaw_magnitude
            elif k == 'u':
                if yaw_magnitude + 2 <= 100:
                    yaw_magnitude += 2
            elif k == 'j':
                if yaw_magnitude - 2 >= 0:
                    yaw_magnitude -= 2
            elif k == 'z':
                up_down = up_down_magnitude
            elif k == 'x':
                up_down = -up_down_magnitude
            elif k == 't':
                if up_down_magnitude + 2 <= 100:
                    up_down_magnitude += 2
            elif k == 'g':
                if up_down_magnitude - 2 >= 0:
                    up_down_magnitude -= 2
            elif k == 'a':
                left_right = -left_right_magnitude
            elif k == 'd':
                left_right = left_right_magnitude
            elif k == 'y':
                if left_right_magnitude + 2 <= 100:
                    left_right_magnitude += 2
            elif k == 'h':
                if left_right_magnitude - 2 >= 0:
                    left_right_magnitude -= 2
            elif k == 'c':
                if current_camera == Tello.CAMERA_FORWARD:
                    current_camera = Tello.CAMERA_DOWNWARD
                    print("📷 Switched to bottom camera")
                else:
                    current_camera = Tello.CAMERA_FORWARD
                    print("📷 Switched to front camera")
                tello.set_video_direction(current_camera)
                time.sleep(0.5)
    except AttributeError:
        pass

# Key release handler
def on_release(key):
    global left_right, up_down, yaw, forward_backward
    try:
        k = key.char.lower()
        with lock:
            if k in 'eq':
                yaw = 0
            if k in 'ws':
                forward_backward = 0
            elif k in 'zx':
                up_down = 0
            elif k in 'ad':
                left_right = 0
    except AttributeError:
        pass

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# Setup matplotlib figure
fig, ax = plt.subplots()
frame = frame_read.frame
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
im = ax.imshow(frame_rgb)
ax.axis("off")
fig.canvas.manager.set_window_title("Tello Camera + Green Detection")

# For RGB on mouse hover
latest_rgb_frame = frame_rgb.copy()
def on_mouse_move(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < latest_rgb_frame.shape[0] and 0 <= x < latest_rgb_frame.shape[1]:
            r, g, b = latest_rgb_frame[y, x]
            fig.canvas.manager.set_window_title(f"RGB at ({x},{y}): R={r}, G={g}, B={b}")

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Frame update with green detection
def update(*args):
    global latest_rgb_frame
    frame = frame_read.frame

    # 🔄 Convert BGR to HSV for masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 🎯 Find green object centroid
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)  # draw in BGR

    # 🔁 Now convert to RGB **after** drawing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    latest_rgb_frame = frame_rgb.copy()
    im.set_array(frame_rgb)
    return [im]


# Start animation loop
ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

print("🕹️ Drone control ready. Press 'C' to switch cameras. Close window to land.")
try:
    plt.show()
finally:
    print("🛬 Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.end()
