import cv2
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from djitellopy import Tello

# === PID Controller ===
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

# === PID Gains ===
pid_yaw = PID(Kp=-0.38, Ki=0.0, Kd=0.0)
pid_z   = PID(Kp=0.16, Ki=0.0, Kd=0.0)
pid_fb  = PID(Kp=-1.0,  Ki=0.0, Kd=0.0)
target_distance_cm = 30.0

# === Tello Setup ===
tello = Tello(host="192.168.12.6")
tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()
print("🛫 Taking off...")
tello.takeoff()
tello.send_rc_control(0, 0, 0, 0)
time.sleep(2)  # let it stabilize a bit
tello.set_video_direction(Tello.CAMERA_DOWNWARD)

# === Control Variables ===
yaw = up_down = forward_backward = 0
lock = threading.Lock()

# === RC Loop ===
def rc_thread():
    while True:
        with lock:
            tello.send_rc_control(0, forward_backward, up_down, yaw)
        time.sleep(0.05)

threading.Thread(target=rc_thread, daemon=True).start()

# === ArUco Dictionary ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# === Visualization ===
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
ax.axis("off")
fig.canvas.manager.set_window_title("Tello ArUco Tracker")

# === State ===
follow_mode = True
marker_id_to_follow = 17  # Change if you want to track another ID
last_time = time.time()

# === Update Loop ===
def update(_):
    global yaw, up_down, forward_backward, last_time
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    corners, ids, _ = detector.detectMarkers(gray)

    if follow_mode and ids is not None and marker_id_to_follow in ids:
        idx = list(ids.flatten()).index(marker_id_to_follow)
        pts = corners[idx][0]

        # Center of the ArUco marker
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))

        # Estimate a pseudo-distance using size
        side_length = np.linalg.norm(pts[0] - pts[2])
        distance_cm = (18 * 200) / side_length

        error_x = frame.shape[1] // 2 - center_x
        error_y = frame.shape[0] // 2 - center_y
        error_d = target_distance_cm - distance_cm

        dt = time.time() - last_time
        last_time = time.time()

        with lock:
            yaw = int(np.clip(pid_yaw.compute(error_x, dt), -50, 50))
            up_down = int(np.clip(pid_z.compute(error_y, dt), -50, 50))
            forward_backward = int(np.clip(pid_fb.compute(error_d, dt), -50, 50))

        # Draw marker
        cv2.aruco.drawDetectedMarkers(frame, [pts])
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(frame,
                    f"ErrX:{error_x:.1f} ErrY:{error_y:.1f} ErrD:{error_d:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        with lock:
            yaw = up_down = forward_backward = 0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im.set_array(rgb_frame)
    return [im]

# === Start Visualization ===
ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)

print("🕹️ Tello ArUco Tracker running. Press Ctrl+C or close window to land.")
try:
    plt.show()
finally:
    print("🛬 Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.end()
