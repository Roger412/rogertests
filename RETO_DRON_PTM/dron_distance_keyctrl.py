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
import matplotlib.gridspec as gridspec
logging.getLogger('djitellopy').setLevel(logging.CRITICAL)
# === Drone Setup ===
tello = Tello(host="192.168.12.6")
# tello = Tello()
tello.connect()
tello.streamon()

current_camera = Tello.CAMERA_FORWARD
tello.set_video_direction(current_camera)
frame_read = tello.get_frame_read()

# Detection defaults — no sliders
LOWER_GREEN = np.array([35, 40, 40], dtype=np.uint8)
UPPER_GREEN = np.array([85,255,255], dtype=np.uint8)

DETECT_PARAMS = {
  "param1": 80,
  "param2": 20,
  "min_radius": 20,
  "max_radius": 200,
  "min_area": 300,
  "max_area": 10000,
  "circularity_threshold": 0.7
}


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

frame = frame_read.frame


# === Main Display Figure ===
fig_main = plt.figure("Tello View", figsize=(8, 6))  # Camera feed
ax_image = fig_main.add_subplot(1, 2, 1)
ax_mask = fig_main.add_subplot(1, 2, 2)

ax_image.axis("off")
ax_mask.axis("off")
ax_image.set_title("Drone Camera View")
ax_mask.set_title("Binary Green Mask")

im = ax_image.imshow(frame)
fig_sliders = plt.figure("All Sliders", figsize=(9, 8))  # Combined sliders

im = ax_image.imshow(frame)
mask_im = ax_mask.imshow(np.zeros_like(frame[:, :, 0]), cmap='gray')


# === Detection Sliders ===
# fig_det = plt.figure("Detection Sliders", figsize=(3, 7))
# fig_pid1 = plt.figure("PID Yaw/Z Sliders", figsize=(4, 7))
# fig_pid2 = plt.figure("PID FB + Distance", figsize=(4, 7))

# Column 1: Detection sliders
s_param1 = Slider(fig_sliders.add_axes([0.05, 0.86, 0.2, 0.03]), "P1", 1, 200, valinit=100)
s_param2 = Slider(fig_sliders.add_axes([0.05, 0.80, 0.2, 0.03]), "P2", 1, 100, valinit=50)
s_minR   = Slider(fig_sliders.add_axes([0.05, 0.74, 0.2, 0.03]), "MinR", 1, 200, valinit=40)
s_maxR   = Slider(fig_sliders.add_axes([0.05, 0.68, 0.2, 0.03]), "MaxR", 1, 300, valinit=150)
s_minA   = Slider(fig_sliders.add_axes([0.05, 0.62, 0.2, 0.03]), "MinA", 0, 10000, valinit=300)
s_maxA   = Slider(fig_sliders.add_axes([0.05, 0.56, 0.2, 0.03]), "MaxA", 0, 20000, valinit=10000)
s_circ   = Slider(fig_sliders.add_axes([0.05, 0.50, 0.2, 0.03]), "Circ", 0.0, 1.0, valinit=0.7)
s_hmin = Slider(fig_sliders.add_axes([0.05, 0.44, 0.2, 0.03]), 'H Min', 0, 179, valinit=35)
s_hmax = Slider(fig_sliders.add_axes([0.05, 0.38, 0.2, 0.03]), 'H Max', 0, 179, valinit=85)
s_smin = Slider(fig_sliders.add_axes([0.05, 0.32, 0.2, 0.03]), 'S Min', 0, 255, valinit=40)
s_smax = Slider(fig_sliders.add_axes([0.05, 0.26, 0.2, 0.03]), 'S Max', 0, 255, valinit=255)
s_vmin = Slider(fig_sliders.add_axes([0.05, 0.20, 0.2, 0.03]), 'V Min', 0, 255, valinit=40)
s_vmax = Slider(fig_sliders.add_axes([0.05, 0.14, 0.2, 0.03]), 'V Max', 0, 255, valinit=255)

# Column 2: PID Yaw/Z
slider_kp_yaw = Slider(fig_sliders.add_axes([0.38, 0.86, 0.2, 0.03]), 'Kp Yaw', 0.0, 10.0, valinit=pid_yaw.Kp)
slider_ki_yaw = Slider(fig_sliders.add_axes([0.38, 0.80, 0.2, 0.03]), 'Ki Yaw', 0.0, 1.0, valinit=pid_yaw.Ki)
slider_kd_yaw = Slider(fig_sliders.add_axes([0.38, 0.74, 0.2, 0.03]), 'Kd Yaw', 0.0, 1.0, valinit=pid_yaw.Kd)

slider_kp_z = Slider(fig_sliders.add_axes([0.38, 0.68, 0.2, 0.03]), 'Kp Z', 0.0, 10.0, valinit=pid_z.Kp)
slider_ki_z = Slider(fig_sliders.add_axes([0.38, 0.62, 0.2, 0.03]), 'Ki Z', 0.0, 1.0, valinit=pid_z.Ki)
slider_kd_z = Slider(fig_sliders.add_axes([0.38, 0.56, 0.2, 0.03]), 'Kd Z', 0.0, 1.0, valinit=pid_z.Kd)

# Column 3: PID FB + Distance
slider_kp_fb = Slider(fig_sliders.add_axes([0.70, 0.86, 0.2, 0.03]), 'Kp FB', 0.0, 10.0, valinit=pid_fb.Kp)
slider_ki_fb = Slider(fig_sliders.add_axes([0.70, 0.80, 0.2, 0.03]), 'Ki FB', 0.0, 1.0, valinit=pid_fb.Ki)
slider_kd_fb = Slider(fig_sliders.add_axes([0.70, 0.74, 0.2, 0.03]), 'Kd FB', 0.0, 1.0, valinit=pid_fb.Kd)
slider_target_distance = Slider(fig_sliders.add_axes([0.70, 0.68, 0.2, 0.03]), 'Target Dist', 5.0, 50.0, valinit=target_distance_cm)



latest_rgb_frame = frame.copy()
def on_mouse_move(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < latest_rgb_frame.shape[0] and 0 <= x < latest_rgb_frame.shape[1]:
            r, g, b = latest_rgb_frame[y, x]
            fig_main.canvas.manager.set_window_title(f"RGB at ({x},{y}): R={r}, G={g}, B={b}")
fig_main.canvas.mpl_connect('motion_notify_event', on_mouse_move)

def detect_green_circles(image,
                         lower_hsv, upper_hsv,
                         param1, param2,
                         min_radius, max_radius,
                         min_area, max_area,
                         circularity_threshold):
    """
    Returns:
      - image with drawn detections
      - binary mask
      - list of (x,y,r) for each accepted circle  
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # morphology from script #1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    blurred = cv2.GaussianBlur(mask, (5,5), 1)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.0, minDist=30,
        param1=param1, param2=param2,
        minRadius=int(min_radius), maxRadius=int(max_radius)
    )

    accepted = []
    if circles is not None:
        for x,y,r in np.round(circles[0]).astype(int):
            roi = mask[y-r:y+r, x-r:x+r]
            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts: 
                continue
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            circ = 4*np.pi*area/(cv2.arcLength(c,True)**2 + 1e-5)
            if area<min_area or area>max_area or circ<circularity_threshold:
                continue
            # draw
            cv2.circle(image, (x,y), r, (0,255,0), 2)
            dist = (18*200)/(2*r)
            cv2.putText(image, f"{2*r}px|{dist:.1f}cm",
                        (x-r, y-r-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            accepted.append((x,y,r))
    return image, mask, accepted

# === Frame Update Function ===
def update(*args):
    global latest_rgb_frame, last_time, target_distance_cm
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))

    # HSV threshold sliders
    lower_green[:] = [s_hmin.val, s_smin.val, s_vmin.val]
    upper_green[:] = [s_hmax.val, s_smax.val, s_vmax.val]

    # === Detection Parameters ===
    p1    = s_param1.val
    p2    = s_param2.val
    minR  = s_minR.val
    maxR  = s_maxR.val
    minA  = s_minA.val
    maxA  = s_maxA.val
    circT = s_circ.val

    # === Circle Detection (always)
    result, mask = detect_green_circles(
        frame.copy(), lower_green, upper_green,
        p1, p2, minR, maxR, minA, maxA, circT
    )

    # === PID update (always)
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

    # === Follow logic if active
    if follow_mode:
        blurred = cv2.GaussianBlur(mask, (5, 5), 1)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.0, minDist=30,
            param1=int(p1), param2=int(p2),
            minRadius=int(minR), maxRadius=int(maxR)
        )

        if circles is not None:
            x, y, r = np.round(circles[0][0]).astype(int)
            diameter = 2 * r
            distance_cm = (18 * 200) / diameter
            error_x = x - frame.shape[1] // 2
            error_y = frame.shape[0] // 2 - y
            error_d = distance_cm - target_distance_cm

            dt = time.time() - last_time
            last_time = time.time()

            with lock:
                yaw = int(np.clip(pid_yaw.compute(error_x, dt), -50, 50))
                up_down = int(np.clip(pid_z.compute(error_y, dt), -50, 50))
                forward_backward = int(np.clip(pid_fb.compute(error_d, dt), -50, 50))

            print(f"🎯 X: {x}, Y: {y}, D: {distance_cm:.1f}cm, "
                  f"ErrX: {error_x}, ErrD: {error_d:.2f}")
        else:
            with lock:
                yaw = 0
                up_down = 0
                forward_backward = 0

    # === Update GUI ===
    latest_rgb_frame = result.copy()
    im.set_array(result)
    mask_im.set_array(mask)
    return [im, mask_im]

fig_main.set_size_inches(960 / fig_main.dpi, 1040 / fig_main.dpi)
ani = animation.FuncAnimation(fig_main, update, interval=50, cache_frame_data=False)


print("🕹️ Drone control ready. Press 'C' to switch cameras. Close window to land.")
try:
    plt.show()
finally:
    print("🛬 Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.end()
