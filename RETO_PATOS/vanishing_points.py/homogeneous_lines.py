import cv2
import numpy as np

def to_homogeneous(pt):
    return np.array([pt[0], pt[1], 1.0])

def compute_line(p1, p2):
    return np.cross(to_homogeneous(p1), to_homogeneous(p2))

# Load image
img = cv2.imread("RETO_PATOS/frame_00888.jpg")
img_hough = img.copy()
img_homogeneous = np.ones_like(img) * 255  # white canvas

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 20, 100, apertureSize=3)

# Detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
line_endpoints = []

if lines is not None:
    for r_theta in lines:
        r, theta = r_theta[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        # Draw on original
        cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save endpoints for later
        line_endpoints.append(((x1, y1), (x2, y2)))

# Draw the same lines using homogeneous form
h, w = img.shape[:2]

for p1, p2 in line_endpoints:
    l = compute_line(p1, p2)
    a, b, c = l

    pts = []

    # Intersect with left and right image borders
    for x in [0, w-1]:
        if b != 0:
            y = int((-a * x - c) / b)
            if 0 <= y < h:
                pts.append((x, y))

    # Intersect with top and bottom image borders
    for y in [0, h-1]:
        if a != 0:
            x = int((-b * y - c) / a)
            if 0 <= x < w:
                pts.append((x, y))

    if len(pts) >= 2:
        cv2.line(img_homogeneous, pts[0], pts[1], (255, 0, 0), 2)

# Show both results side by side
cv2.imshow("Hough Lines (Direct)", img_hough)
cv2.imshow("Homogeneous Lines (From a,b,c)", img_homogeneous)
cv2.waitKey(0)
cv2.destroyAllWindows()
