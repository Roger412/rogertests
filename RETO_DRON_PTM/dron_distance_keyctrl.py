import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from djitellopy import Tello
from pynput import keyboard
import threading
import time
import cv2
import numpy as np
import logging
logging.getLogger('djitellopy').setLevel(logging.CRITICAL)
# === Drone Setup ===
# tello = Tello(host="192.168.12.6")
tello = Tello()
tello.connect()
tello.streamon()

current_camera = Tello.CAMERA_FORWARD
tello.set_video_direction(current_camera)
frame_read = tello.get_frame_read()


# === Follow mode toggle ===
follow_mode = False

# === PID class ===
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

# === PID Controllers for follow mode ===
pid_yaw = PID(Kp=0.6, Ki=0.15, Kd=0.0)
pid_z = PID(Kp=0.6, Ki=0.15, Kd=0.0)
pid_fb = PID(Kp=0.4, Ki=0.2, Kd=0.0)  # Add near the top if not already defined
target_distance_cm = 12.0            # Desired distance
last_time = time.time()


# === Control Variables ===
forward_backward = left_right = up_down = yaw = 0
forward_backward_magnitude = left_right_magnitude = up_down_magnitude = yaw_magnitude = 40
lock = threading.Lock()

# === HSV Green Range ===
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# === RC Thread ===
def control_loop():
    while True:
        with lock:
            tello.send_rc_control(int(left_right), int(forward_backward), int(up_down), int(yaw))
        time.sleep(0.05)

threading.Thread(target=control_loop, daemon=True).start()

# === Keyboard Handling ===
def on_press(key):
    global left_right, up_down, yaw, forward_backward
    global forward_backward_magnitude, left_right_magnitude, up_down_magnitude, yaw_magnitude
    global current_camera

    try:
        k = key.char.lower()
        with lock:
            if k == 'w': forward_backward = forward_backward_magnitude
            elif k == 's': forward_backward = -forward_backward_magnitude
            elif k == 'r' and forward_backward_magnitude + 2 <= 100: forward_backward_magnitude += 2
            elif k == 'f' and forward_backward_magnitude - 2 >= 0: forward_backward_magnitude -= 2
            elif k == 'e': yaw = yaw_magnitude
            elif k == 'q': yaw = -yaw_magnitude
            elif k == 'u' and yaw_magnitude + 2 <= 100: yaw_magnitude += 2
            elif k == 'j' and yaw_magnitude - 2 >= 0: yaw_magnitude -= 2
            elif k == 'z': up_down = up_down_magnitude
            elif k == 'x': up_down = -up_down_magnitude
            elif k == 't' and up_down_magnitude + 2 <= 100: up_down_magnitude += 2
            elif k == 'g' and up_down_magnitude - 2 >= 0: up_down_magnitude -= 2
            elif k == 'a': left_right = -left_right_magnitude
            elif k == 'd': left_right = left_right_magnitude
            elif k == 'y' and left_right_magnitude + 2 <= 100: left_right_magnitude += 2
            elif k == 'h' and left_right_magnitude - 2 >= 0: left_right_magnitude -= 2
            elif k == 'c':
                current_camera = Tello.CAMERA_DOWNWARD if current_camera == Tello.CAMERA_FORWARD else Tello.CAMERA_FORWARD
                print("📷 Switched camera.")
                tello.set_video_direction(current_camera)
                time.sleep(0.5)
            elif k == 'k':
                print("🛫 Takeoff triggered.")
                tello.takeoff()
            elif k == 'l':
                print("🛬 Land triggered.")
                tello.land()
            elif k == 'p':
                global follow_mode
                follow_mode = not follow_mode
                print(f"🎯 Follow mode {'ON' if follow_mode else 'OFF'}")

    except AttributeError:
        pass

def on_release(key):
    global left_right, up_down, yaw, forward_backward
    try:
        k = key.char.lower()
        with lock:
            if k in 'eq': yaw = 0
            if k in 'ws': forward_backward = 0
            elif k in 'zx': up_down = 0
            elif k in 'ad': left_right = 0
    except AttributeError:
        pass

keyboard.Listener(on_press=on_press, on_release=on_release).start()

# === Matplotlib Setup ===
fig, ax = plt.subplots()
frame = frame_read.frame
im = ax.imshow(frame)
# Adjust figure to fit sliders
plt.subplots_adjust(left=0.25, bottom=0.4)

# Create slider axes
ax_kp_yaw = plt.axes([0.25, 0.33, 0.65, 0.03])
ax_ki_yaw = plt.axes([0.25, 0.29, 0.65, 0.03])
ax_kd_yaw = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_kp_z = plt.axes([0.25, 0.21, 0.65, 0.03])
ax_ki_z = plt.axes([0.25, 0.17, 0.65, 0.03])
ax_kd_z = plt.axes([0.25, 0.13, 0.65, 0.03])
# === Forward/Backward PID sliders ===
ax_kp_fb = plt.axes([0.25, 0.09, 0.65, 0.03])
ax_ki_fb = plt.axes([0.25, 0.05, 0.65, 0.03])
ax_kd_fb = plt.axes([0.25, 0.01, 0.65, 0.03])
ax_target_distance = plt.axes([0.05, 0.5, 0.0225, 0.35])  # Vertical slider

# Create sliders
slider_kp_yaw = Slider(ax_kp_yaw, 'Kp Yaw', 0.0, 10.0, valinit=pid_yaw.Kp)
slider_ki_yaw = Slider(ax_ki_yaw, 'Ki Yaw', 0.0, 1.0, valinit=pid_yaw.Ki)
slider_kd_yaw = Slider(ax_kd_yaw, 'Kd Yaw', 0.0, 1.0, valinit=pid_yaw.Kd)
slider_kp_z = Slider(ax_kp_z, 'Kp Z', 0.0, 10.0, valinit=pid_z.Kp)
slider_ki_z = Slider(ax_ki_z, 'Ki Z', 0.0, 1.0, valinit=pid_z.Ki)
slider_kd_z = Slider(ax_kd_z, 'Kd Z', 0.0, 1.0, valinit=pid_z.Kd)
slider_kp_fb = Slider(ax_kp_fb, 'Kp FB', 0.0, 10.0, valinit=pid_fb.Kp)
slider_ki_fb = Slider(ax_ki_fb, 'Ki FB', 0.0, 1.0, valinit=pid_fb.Ki)
slider_kd_fb = Slider(ax_kd_fb, 'Kd FB', 0.0, 1.0, valinit=pid_fb.Kd)
slider_target_distance = Slider(ax_target_distance, 'Target\nDist', 5.0, 50.0, valinit=target_distance_cm, orientation='vertical')

ax.axis("off")
fig.canvas.manager.set_window_title("Tello Circle Detection")

latest_rgb_frame = frame.copy()
def on_mouse_move(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < latest_rgb_frame.shape[0] and 0 <= x < latest_rgb_frame.shape[1]:
            r, g, b = latest_rgb_frame[y, x]
            fig.canvas.manager.set_window_title(f"RGB at ({x},{y}): R={r}, G={g}, B={b}")
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# === Green Circle Detection ===
def detect_green_circles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=50, minRadius=40, maxRadius=150)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if mask[y, x] > 0:
                diameter = 2 * r
                distance_cm = (6.7 * 200) / diameter
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.putText(image, f"{diameter}px | {distance_cm:.1f}cm",
                            (x - r, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# === Frame Update Function ===
def update(*args):
    global latest_rgb_frame, last_time, target_distance_cm
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))
    frame = detect_green_circles(frame)

    # === Update PID gains from trackbars ===
    pid_yaw.Kp = slider_kp_yaw.val
    pid_yaw.Ki = slider_ki_yaw.val
    pid_yaw.Kd = slider_kd_yaw.val

    pid_z.Kp = slider_kp_z.val
    pid_z.Ki = slider_ki_z.val
    pid_z.Kd = slider_kd_z.val

    pid_fb.Kp = slider_kp_fb.val
    pid_fb.Ki = slider_ki_fb.val
    pid_fb.Kd = slider_kd_fb.val
    target_distance_cm = slider_target_distance.val


    if follow_mode:
        global yaw, up_down, forward_backward
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=100, param2=50, minRadius=40, maxRadius=150)

        dt = time.time() - last_time
        last_time = time.time()

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])
            x, y, r = largest_circle
            cx, cy = x, y

            diameter_px = 2 * r
            distance_cm = (6.7 * 200) / diameter_px  # Focal * Real_D / Diameter
            error_x = cx - frame.shape[1] // 2
            error_y = frame.shape[0] // 2 - cy
            error_d = distance_cm - target_distance_cm

            with lock:
                yaw = int(np.clip(pid_yaw.compute(error_x, dt), -50, 50))
                up_down = int(np.clip(pid_z.compute(error_y, dt), -50, 50))
                forward_backward = int(np.clip(pid_fb.compute(error_d, dt), -50, 50))
                forward_backward = forward_backward * 8
            # Debug overlay
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{distance_cm:.1f}cm", (cx - r, cy - r - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            print(f"FB Error: {error_d:.2f}, FB Output: {forward_backward} Measured Distance: {distance_cm:.2f} cm, Target: {target_distance_cm}")

        else:
            with lock:
                yaw = 0
                up_down = 0
                forward_backward = 0



    latest_rgb_frame = frame.copy()
    im.set_array(frame)
    return [im]

fig.set_size_inches(960 / fig.dpi, 1040 / fig.dpi)
ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

print("🕹️ Drone control ready. Press 'C' to switch cameras. Close window to land.")
try:
    plt.show()
finally:
    print("🛬 Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.end()
