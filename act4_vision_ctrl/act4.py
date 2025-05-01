import matplotlib.pyplot as plt
import matplotlib.animation as animation
from djitellopy import Tello
import cv2
import numpy as np

# Connect to Tello and start stream
tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()

# Set up the figure and image object
fig, ax = plt.subplots()
frame = tello.get_frame_read().frame
im = ax.imshow(np.zeros_like(frame))  # dummy init

# Remove axis ticks
ax.axis("off")

# Define HSV bounds for light green
lower_green = np.array([35, 100, 100])   # adjust as needed
upper_green = np.array([85, 255, 255])

def update(*args):
    frame = tello.get_frame_read().frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    im.set_array(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return [im]

# Start animation
ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

try:
    plt.show()
finally:
    tello.send_rc_control(0, 0, 0, 0)
    tello.land()
    tello.streamoff()
    tello.end()
