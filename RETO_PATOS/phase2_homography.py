import numpy as np
import cv2
import os

clicked_points = []

def click_event(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: ({x}, {y})")
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 corners of one tile", param)

# Cargar imagen para selección
img_path = "RETO_PATOS/frames_key/frame_0000.jpg"
img = cv2.imread(img_path)
img_copy = img.copy()

cv2.imshow("Select 4 corners of one tile", img_copy)
cv2.setMouseCallback("Select 4 corners of one tile", click_event, img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:
    raise ValueError("You must click exactly 4 points!")

pts_img = np.float32(clicked_points)
pts_real = np.float32([
    [0, 0],
    [30, 0],
    [30, 30],
    [0, 30],
])

H, _ = cv2.findHomography(pts_img, pts_real)

print(pts_img)
print(pts_real)

print(H)

# Guardar matriz
np.save("RETO_PATOS/homography_matrix/homography_matrix.npy", H)
print("\n✅ Homography matrix saved as 'homography_matrix.npy'")
