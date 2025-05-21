import matplotlib.pyplot as plt
import matplotlib.animation as animation
from djitellopy import Tello
import cv2
import numpy as np
import time

# ──────────────── PID Controller ────────────────
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

# ──────────────── Drone Setup ────────────────
tello = Tello(host="192.168.12.6")
tello.connect()
tello.streamon() 
tello.takeoff()

# Get initial frame
frame = tello.get_frame_read().frame
h, w, _ = frame.shape
center_x = w // 2
center_y = h // 2

fig, ax = plt.subplots()
im = ax.imshow(frame)
ax.axis("off")

# HSV bounds for light green
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# PID controllers
pid_yaw = PID(Kp=0.2, Ki=0.01, Kd=0.0)  # left/right (yaw)
pid_z   = PID(Kp=0.2, Ki=0.01, Kd=0.0)  # up/down (altitude)
last_time = time.time()

# RGB inspector
latest_frame = frame.copy()

def on_mouse_move(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < latest_frame.shape[0] and 0 <= x < latest_frame.shape[1]:
            b, g, r = latest_frame[y, x]
            fig.canvas.manager.set_window_title(
                f"RGB at ({x},{y}): R={r}, G={g}, B={b}"
            )

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# ──────────────── Frame Update ────────────────
def update(*args):
    global latest_frame, last_time
    frame = tello.get_frame_read().frame
    latest_frame = frame.copy()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    M = cv2.moments(mask)
    dt = time.time() - last_time
    last_time = time.time()

    yaw_cmd = z_cmd = 0

    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        error_x = cx - center_x
        error_y = center_y - cy

        yaw_cmd = int(np.clip(pid_yaw.compute(error_x, dt), -50, 50))
        z_cmd   = int(np.clip(pid_z.compute(error_y, dt), -50, 50))

        # Draw red dot on centroid
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    else:
        yaw_cmd = z_cmd = 0  # no target

    # Send control command
    tello.send_rc_control(0, 0, z_cmd, yaw_cmd)

    im.set_array(frame)
    return [im]

# ──────────────── Start Stream ────────────────
ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

try:
    plt.show()
finally:
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    tello.end()
