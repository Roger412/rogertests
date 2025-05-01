import time
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
from djitellopy import Tello

# === Telemetry Printing Thread ===
def telemetry_loop(tello: Tello, stop_flag):
    while not stop_flag["stop"]:
        state = tello.get_current_state()
        battery = state.get("bat", "?")
        height = state.get("h", "?")
        temp_l = state.get("templ", "?")
        temp_h = state.get("temph", "?")
        print(f"🔋 Battery: {battery}% | 📏 Height: {height} cm | 🌡 Temp: {temp_l}-{temp_h}°C")
        time.sleep(0.5)

# === Start drone and stream ===
tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()

# === Prepare circular trajectory ===
R = 50
omega = 2 * math.pi / 5
duration = 6
dt = 0.05
n_steps = int(duration / dt)

trajectory = []
for i in range(n_steps):
    t = i * dt
    vx = int(-R * math.sin(omega * t))
    vy = int(R * math.cos(omega * t))
    trajectory.append((vx, vy))

# === Prepare live frame read ===
frame_reader = tello.get_frame_read()

fig, ax = plt.subplots()
im = ax.imshow(frame_reader.frame)
ax.axis("off")

# === Animate video stream ===
def update(*args):
    im.set_array(frame_reader.frame)
    return [im]

ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

# === Start telemetry in parallel ===
stop_flag = {"stop": False}
telemetry_thread = Thread(target=telemetry_loop, args=(tello, stop_flag))
telemetry_thread.start()

# === Run drone trajectory in another thread ===
def circular_trajectory():
    time.sleep(1)
    for vx, vy in trajectory:
        tello.send_rc_control(vx, vy, 0, 0)
        time.sleep(dt)
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    tello.land()
    tello.streamoff()
    stop_flag["stop"] = True
    telemetry_thread.join()
    tello.end()
    plt.close()

flight_thread = Thread(target=circular_trajectory)
flight_thread.start()

# === Start video display (blocking) ===
plt.show()
