from djitellopy import Tello

# Replace with IP discovered in step 3
tello = Tello(host="192.168.12.6")
tello.connect()

print("Battery level:", tello.get_battery())