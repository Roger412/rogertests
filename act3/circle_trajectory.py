import time
from djitellopy import Tello
from threading import Thread

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

try:
    # Move right and forward with slight yaw to simulate a circle
    tello.send_rc_control(30, 30, 0, 20)
    time.sleep(8)

    # Stop motion
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
finally:
    stop_flag["stop"] = True
    telemetry_thread.join()

    tello.land()
    tello.end()
