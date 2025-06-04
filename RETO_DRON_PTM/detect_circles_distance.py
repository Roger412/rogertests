import cv2
import numpy as np


# Caso para aplicar regla de 3
# Diametro circulo: 6.7 cm
# Pixeles en la imagen: 200
# Distancia de la cámara 18 cm


def detect_green_circles(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green color range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Mask for green
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    # Detect circles with Hough Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,param1=100, param2=50, minRadius=40, maxRadius=150)
    diameters = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Only keep circles that are green in the original mask
            if mask[y, x] > 0:
                diameter = 2 * r
                diameters.append((diameter, (x, y, r)))

        # Sort by diameter (descending)
        diameters.sort(reverse=True)

        for i, (diameter, (x, y, r)) in enumerate(diameters):
            # Estimate distance based on calibration: 200 px <-> 18 cm
            distance_cm = (18 * 200) / diameter  # 6.7cm is real size

            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            text = f"{diameter}px | {distance_cm:.1f}cm"
            cv2.putText(image, text, (x - r, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    return image, diameters

# ==== Webcam or image file ====
use_webcam = True  # set to False to use image file

if use_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img, sorted_circles = detect_green_circles(frame)
        cv2.imshow("Green Circle Detection", result_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
else:
    image = cv2.imread("your_image.jpg")  # Change to your image path
    result_img, sorted_circles = detect_green_circles(image)
    cv2.imshow("Green Circle Detection", result_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
