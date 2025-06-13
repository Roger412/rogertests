import cv2
import numpy as np
from djitellopy import Tello
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


mouse_x, mouse_y = None, None


# === Circle detection ===
def detect_green_circles(image, lower_hsv, upper_hsv,
                         param1, param2, min_radius, max_radius, min_area, max_area):
    global s_circ
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # === Contour filtering ===
    # Blur mask to prep for Hough
    blurred = cv2.GaussianBlur(mask, (5, 5), 1)

    # Apply HoughCircles on the blurred binary mask
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=30,
        param1=param1,
        param2=param2,
        minRadius=int(min_radius),
        maxRadius=int(max_radius)
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            roi = mask[y - r:y + r, x - r:x + r]
            cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (perimeter**2 + 1e-5)
            if circularity < s_circ.val:
                continue

            # ✅ draw detection
            diameter = 2 * r
            distance_cm = (18 * 200) / diameter
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            text = f"{diameter}px | {distance_cm:.1f}cm"
            cv2.putText(image, text, (x - r, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    return image, mask


# === Tello setup ===
tello = Tello(host="192.168.12.6")
tello.connect()
tello.streamon()
tello.set_video_direction(Tello.CAMERA_DOWNWARD)

print("📡 Connected to Tello. Close the Matplotlib window to stop.")

# === Matplotlib Setup ===
frame = tello.get_frame_read().frame
frame = cv2.resize(frame, (640, 480))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

import matplotlib.gridspec as gridspec

# === Resize and create figure ===
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(3, 4, height_ratios=[5, 0.2, 0.2])
ax_frame = plt.subplot(gs[0, 0:2])
ax_mask = plt.subplot(gs[0, 2:4])

ax_frame.set_title("Camera View")
ax_mask.set_title("HSV Mask")
ax_frame.axis("off")
ax_mask.axis("off")

im_frame = ax_frame.imshow(frame)
im_mask = ax_mask.imshow(np.zeros_like(frame))


# === Slider axes and sliders ===
def slider_ax(col, row):
    left = 0.1 if col == 0 else 0.55
    bottom = 0.25 - row * 0.045
    return plt.axes([left, bottom, 0.35, 0.03])


s_hmin = Slider(slider_ax(0, 0), 'H Min', 0, 179, valinit=35, valstep=1)
s_hmax = Slider(slider_ax(0, 1), 'H Max', 0, 179, valinit=85, valstep=1)
s_smin = Slider(slider_ax(0, 2), 'S Min', 0, 255, valinit=40, valstep=1)
s_smax = Slider(slider_ax(0, 3), 'S Max', 0, 255, valinit=255, valstep=1)
s_vmin = Slider(slider_ax(0, 4), 'V Min', 0, 255, valinit=40, valstep=1)
s_vmax = Slider(slider_ax(0, 5), 'V Max', 0, 255, valinit=255, valstep=1)

s_param1 = Slider(slider_ax(1, 0), 'Hough Param1', 1, 200, valinit=80, valstep=1)
s_param2 = Slider(slider_ax(1, 1), 'Hough Param2', 1, 100, valinit=20, valstep=1)
s_minR   = Slider(slider_ax(1, 2), 'Min Radius', 1, 200, valinit=20, valstep=1)
s_maxR   = Slider(slider_ax(1, 3), 'Max Radius', 1, 300, valinit=200, valstep=1)
s_minA   = Slider(slider_ax(1, 4), 'Min Area', 100, 10000, valinit=300, valstep=50)
s_maxA   = Slider(slider_ax(1, 5), 'Max Area', 100, 10000, valinit=10000, valstep=50)
s_circ = Slider(slider_ax(1, 6), 'Circularity Min', 0.0, 1.0, valinit=0.7, valstep=0.01)


# === Animation update function ===
def update(frame_num):
    global mouse_x, mouse_y

    frame = tello.get_frame_read().frame
    frame = cv2.resize(frame, (640, 480))

    lower = np.array([s_hmin.val, s_smin.val, s_vmin.val], dtype=np.uint8)
    upper = np.array([s_hmax.val, s_smax.val, s_vmax.val], dtype=np.uint8)

    # HSV + mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Circle detection
    result, mask = detect_green_circles(
        frame.copy(), lower, upper,
        s_param1.val, s_param2.val,
        s_minR.val, s_maxR.val,
        s_minA.val, s_maxA.val
    )
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


    # HSV value at mouse
    if mouse_x is not None and mouse_y is not None:
        if 0 <= mouse_y < hsv.shape[0] and 0 <= mouse_x < hsv.shape[1]:
            h, s, v = hsv[mouse_y, mouse_x]
            text = f'HSV: ({h}, {s}, {v})'
            cv2.putText(result, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Update matplotlib images
    im_frame.set_array(result)
    im_mask.set_array(mask_rgb)


def on_mouse_move(event):
    global mouse_x, mouse_y
    if event.xdata and event.ydata:
        mouse_x = int(event.xdata)
        mouse_y = int(event.ydata)

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)


ani = FuncAnimation(fig, update, interval=50)
plt.show()

# === Cleanup ===
tello.streamoff()
tello.end()
