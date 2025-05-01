import math
import matplotlib.pyplot as plt
import time
from djitellopy import Tello
from threading import Thread

# Parameters
R = 50
omega = 2 * math.pi / 5  # one full circle in 5s
duration = 6
dt = 0.05
n_steps = int(duration / dt)

# Function to constantly print telemetry
def telemetry_loop(tello: Tello, stop_flag, log):
    while not stop_flag["stop"]:
        state = tello.get_current_state()
        battery = state.get("bat", "?")
        height = state.get("h", "?")
        temp_l = state.get("templ", "?")
        temp_h = state.get("temph", "?")
        print(f"🔋 Battery: {battery}% | 📏 Height: {height} cm | 🌡 Temp: {temp_l}-{temp_h}°C")
        try:
            x = float(state.get("vgx", 0))
            y = float(state.get("vgy", 0))
            log.append((x, y))
        except ValueError:
            pass
        time.sleep(0.1)

# Precompute smooth acceleration curve (sinusoidal envelope)
def envelope(t_norm):
    return math.sin(math.pi * t_norm)

# Precompute circular trajectory
trajectory = []
for i in range(n_steps):
    t = i * dt
    norm = i / n_steps
    scale = envelope(norm)
    vx = int(-R * math.sin(omega * t) * scale)
    vy = int(R * math.cos(omega * t) * scale)
    trajectory.append((vx, vy))

# Init drone
tello = Tello()
tello.connect()
tello.takeoff()

# Telemetry logging
stop_flag = {"stop": False}
log_data = []
telemetry_thread = Thread(target=telemetry_loop, args=(tello, stop_flag, log_data))
telemetry_thread.start()

time.sleep(1)

try:
    print("🌀 Executing smoothed circular trajectory...")
    for vx, vy in trajectory:
        tello.send_rc_control(vx, vy, 0, 0)
        time.sleep(dt)
finally:
    tello.send_rc_control(0, 0, 0, 0)
    stop_flag["stop"] = True
    telemetry_thread.join()
    tello.land()
    tello.end()

# Plot trajectory
x_vals = [pt[0] for pt in log_data]
y_vals = [pt[1] for pt in log_data]

plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_vals, marker='o', linewidth=1)
plt.title("Drone Velocity Trajectory")
plt.xlabel("Velocity X (cm/s)")
plt.ylabel("Velocity Y (cm/s)")
plt.grid(True)
plt.axis("equal")
plt.show()
