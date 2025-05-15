from djitellopy import Tello
import time

tello = Tello()
tello.connect()

# Replace with your Wi-Fi name and password
ssid = "rogernet"
password = "roger4412"

# Send credentials to the drone
tello.send_command_with_return(f"ap {ssid} {password}")
print("✅ Sent Wi-Fi credentials")

# Wait and reboot
time.sleep(3)
tello.end()
print("🔁 Tello should now reboot and connect to your Wi-Fi")
