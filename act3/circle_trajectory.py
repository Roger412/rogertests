import time
from djitellopy import Tello
from threading import Thread
import math

# Function to constantly print telemetry
def telemetry_loop(tello: Tello, stop_flag):
    while not stop_flag["stop"]:
        state = tello.get_current_state()
        battery = state.get("bat", "?")
        height = state.get("h", "?")
        temp_l = state.get("templ", "?")
        temp_h = state.get("temph", "?")
        print(f"🔋 Battery: {battery}% | 📏 Height: {height} cm | 🌡 Temp: {temp_l}-{temp_h}°C")
        time.sleep(0.5)

# Initialize and take off
tello = Tello()
tello.connect()
tello.takeoff()

print("🌀 Starting circular trajectory...")

# Shared flag to stop telemetry thread
stop_flag = {"stop": False}
telemetry_thread = Thread(target=telemetry_loop, args=(tello, stop_flag))
telemetry_thread.start()

R = 50      # radius scaled to drone speed
omega = 2 * math.pi / 5  # complete circle in 5 seconds
duration = 6             # flight time
dt = 0.05                # timestep (s)
n_steps = int(duration / dt)

trajectory = []
for i in range(n_steps):
    t = i * dt
    vx = int(-R * math.sin(omega * t))
    vy = int(R * math.cos(omega * t))
    trajectory.append((vx, vy))

vx = max(-100, min(100, int(-R * math.sin(omega * t))))
vy = max(-100, min(100, int(R * math.cos(omega * t))))

time.sleep(1)

try:
    for vx, vy in trajectory:
        tello.send_rc_control(vx, vy, 0, 0)
        time.sleep(dt)
finally:
    # Stop motion
    tello.send_rc_control(0, 0, 0, 0)
    stop_flag["stop"] = True
    telemetry_thread.join()
    tello.land()
    tello.end()
