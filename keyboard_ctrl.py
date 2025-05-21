from djitellopy import Tello
from pynput import keyboard
import threading
import time

# Initialize drone
tello = Tello(host="192.168.12.6")
tello.connect()
tello.streamon()
tello.takeoff()

# Velocity variables (range -100 to 100)
left_right = 0    # x-axis (Z/X)
up_down = 0       # z-axis (W/S)
yaw = 0           # yaw (A/D)

lock = threading.Lock()

# Send RC control in a loop
def control_loop():
    while True:
        with lock:
            tello.send_rc_control(int(left_right), 0, int(up_down), int(yaw))
        time.sleep(0.05)

threading.Thread(target=control_loop, daemon=True).start()

# Key press handler
def on_press(key):
    global left_right, up_down, yaw
    try:
        k = key.char.lower()
        with lock:
            if k == 'a':
                yaw = -50
            elif k == 'd':
                yaw = 50
            elif k == 'w':
                up_down = 50
            elif k == 's':
                up_down = -50
            elif k == 'z':
                left_right = -50
            elif k == 'x':
                left_right = 50
    except AttributeError:
        pass

# Key release handler
def on_release(key):
    global left_right, up_down, yaw
    try:
        k = key.char.lower()
        with lock:
            if k in 'ad':
                yaw = 0
            elif k in 'ws':
                up_down = 0
            elif k in 'zx':
                left_right = 0
    except AttributeError:
        pass

# Start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("🕹️ Control: A/D = yaw | W/S = up/down | Z/X = left/right")
print("Press ESC to land and exit...")

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    print("🛬 Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.end()
