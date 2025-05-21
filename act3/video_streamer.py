import matplotlib.pyplot as plt
import matplotlib.animation as animation
from djitellopy import Tello
import cv2

# Connect to Tello and start stream
tello = Tello(host="192.168.12.6")
tello.connect()
tello.streamon()

# Set up the figure and image object
fig, ax = plt.subplots()
frame = tello.get_frame_read().frame
im = ax.imshow(frame)

# Remove axis ticks
ax.axis("off")

def update(*args):
    frame = tello.get_frame_read().frame
    im.set_array(frame)
    return [im]

# Start animation
ani = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)

try:
    plt.show()
finally:
    tello.streamoff()
    tello.end()
