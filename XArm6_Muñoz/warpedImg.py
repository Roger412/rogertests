import cv2
import numpy as np

# Load the image
img = cv2.imread("XArm6_Muñoz/After.png")  # Change path as needed
if img is None:
    print("Image not found.")
    exit()

clicked_points = []

def click_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append([x, y])
        print(f"Point {len(clicked_points)}: ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow("Click Board Corners", img)

# Show image and collect 4 points
cv2.imshow("Click Board Corners", img)
cv2.setMouseCallback("Click Board Corners", click_points)

print("Click the 4 board corners in order: top-left, top-right, bottom-right, bottom-left.")
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(clicked_points) != 4:
    print("Error: You must click exactly 4 points.")
    exit()

board_image = np.array(clicked_points, dtype=np.float32)

# Define real-world corners of the board (mm)
board_real = np.array([
    [0, 0],
    [300, 0],
    [300, 200],
    [0, 200]
], dtype=np.float32)

# Compute homography
H, _ = cv2.findHomography(board_real, board_image)

# Warp image to flatten the board
height, width = 400, 600  # or use int(200*scale), int(300*scale)
warped = cv2.warpPerspective(img, np.linalg.inv(H), (width, height))

cv2.imshow("Rectified Board View", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
