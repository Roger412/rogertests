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
tello = Tello(host="192.168.12.6")
# tello = Tello()
tello.connect()
tello.streamon()

current_camera = Tello.CAMERA_FORWARD
tello.set_video_direction(current_camera)
frame_read = tello.get_frame_read()


# === Follow mode toggle ===
follow_mode = False

# detection defaults
DETECT_PARAMS = {
    "param1": 80,
    "param2": 20,
    "min_radius": 20,
    "max_radius": 200,
    "min_area": 300,
    "max_area": 10000,
    "circularity_threshold": 0.7
}


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
      - list of (x,y,r) accepted circles
    """
    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # morphology cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel, iterations=1)

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
            if not (min_area <= area <= max_area and circ >= circularity_threshold):
                continue

            # draw on original image
            cv2.circle(image, (x,y), r, (0,255,0), 2)
            dist = (18 * 200) / (2*r)
            cv2.putText(image, f"{2*r}px|{dist:.1f}cm",
                        (x-r, y-r-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            accepted.append((x,y,r))

    return image, mask, accepted


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
pid_yaw = PID(Kp=-0.38, Ki=0.0, Kd=0.0)
# pid_z = PID(Kp=0.1, Ki=0.05, Kd=0.0)
# pid_fb = PID(Kp=0.4, Ki=0.05, Kd=0.0)  # Add near the top if not already defined
pid_z = PID(Kp=0.16, Ki=0.0, Kd=0.0)
pid_fb = PID(Kp=-1.0, Ki=0.0, Kd=0.0)  # Add near the top if not already defined
target_distance_cm = 45.0            # Desired distance
last_time = time.time()


# === Control Variables ===
forward_backward = left_right = up_down = yaw = 0
forward_backward_magnitude = left_right_magnitude = up_down_magnitude = yaw_magnitude = 40
lock = threading.Lock()

# === HSV Green Range ===
# right after your imports:
color_mode = 'green'

# existing green range
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# two‐part red range (because red wraps around 0° in HSV)
h, s, v = 176, 136, 218
delta_h = 20
delta_s = 40
delta_v = 40

lower_red1 = np.array([115, 166, 117])
upper_red1 = np.array([168, 255, 240])


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
    global current_camera, color_mode

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
            elif k == 'm':
                # flip between green and red
                color_mode = 'red' if color_mode=='green' else 'green'
                print(f"🔄 Now detecting {color_mode} circles")

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
slider_kp_yaw = Slider(ax_kp_yaw, 'Kp Yaw', -10.0, 10.0, valinit=pid_yaw.Kp)
slider_ki_yaw = Slider(ax_ki_yaw, 'Ki Yaw', -1.0, 1.0, valinit=pid_yaw.Ki)
slider_kd_yaw = Slider(ax_kd_yaw, 'Kd Yaw', -1.0, 1.0, valinit=pid_yaw.Kd)
slider_kp_z = Slider(ax_kp_z, 'Kp Z', -10.0, 10.0, valinit=pid_z.Kp)
slider_ki_z = Slider(ax_ki_z, 'Ki Z', -1.0, 1.0, valinit=pid_z.Ki)
slider_kd_z = Slider(ax_kd_z, 'Kd Z', -1.0, 1.0, valinit=pid_z.Kd)
slider_kp_fb = Slider(ax_kp_fb, 'Kp FB', -10.0, 10.0, valinit=pid_fb.Kp)
slider_ki_fb = Slider(ax_ki_fb, 'Ki FB', -1.0, 1.0, valinit=pid_fb.Ki)
slider_kd_fb = Slider(ax_kd_fb, 'Kd FB', -1.0, 1.0, valinit=pid_fb.Kd)
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


# === Frame Update Function ===
def update(*args):
    global latest_rgb_frame, last_time, target_distance_cm
    global yaw, up_down, forward_backward
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))
    lower = None
    upper = None
    # run the new detector
    if color_mode == 'green':
        lower = lower_green
        upper = upper_green        
    else:
        lower = lower_red1
        upper = upper_red1

    result, mask, circles = detect_green_circles(
        frame.copy(),
        lower, upper,
        **DETECT_PARAMS
    )
    frame = result


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


    if follow_mode and circles:
        # pick the largest or first
        x,y,r = max(circles, key=lambda c: c[2])
        diameter    = 2*r
        distance_cm = (18 * 200)/diameter

        error_x = frame.shape[1]//2 - x
        error_y = frame.shape[0]//2 - y
        error_d = target_distance_cm - distance_cm

        dt = time.time() - last_time
        last_time = time.time()

        with lock:
            if (color_mode == 'green'):
                yaw             = int(np.clip(pid_yaw.compute(error_x, dt), -50, 50))
                up_down         = int(np.clip(pid_z.compute(error_y,   dt), -50, 50))
                forward_backward = int(np.clip(pid_fb.compute(error_d, dt), -50, 50))
            else:
                yaw             = int(np.clip(-pid_yaw.compute(error_x, dt), -50, 50))
                up_down         = int(np.clip(-pid_z.compute(error_y,   dt), -50, 50))
                forward_backward = int(np.clip(-pid_fb.compute(error_d, dt) /2, -50, 50))

            text = f"ErrX:{error_x:.1f}  ErrY:{error_y:.1f}  ErrD:{error_d:.1f}"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        with lock:
            yaw = up_down = forward_backward = 0



    latest_rgb_frame = frame.copy()
    cv2.putText(frame,
            f"Mode: {color_mode}",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255,255,255), 2)
    
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