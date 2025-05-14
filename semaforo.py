import cv2
import numpy as np

def detect_colored_circles(frame):
    output = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        "red": [
            ((0, 100, 100), (10, 255, 255)),
            ((160, 100, 100), (179, 255, 255))  # Red has two ranges in HSV
        ],
        "green": [((40, 70, 70), (80, 255, 255))],
        "yellow": [((20, 100, 100), (30, 255, 255))]
    }

    for color, ranges in color_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask += cv2.inRange(hsv, lower, upper)

        # Blur and apply HoughCircles
        blurred = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=30, param1=100, param2=30,
                                   minRadius=10, maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 255), 2)
                cv2.putText(output, color, (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output

# ---------- MAIN ----------
cap = cv2.VideoCapture(0)  # Or use cv2.imread("image.jpg") for a static image

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_colored_circles(frame)
    cv2.imshow("Detected Circles", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
