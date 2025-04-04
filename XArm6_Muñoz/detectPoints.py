import cv2
import numpy as np

def detect_colored_points(image_path):
    """
    Detects red, blue, and green points in an image using OpenCV.

    Args:
        image_path (str): The path to the input image.

    Returns:
        dict: A dictionary containing lists of detected red, blue, and green points.
              Each point is represented as a tuple (x, y).
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define color ranges
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_blue = np.array([100, 180, 120])
        upper_blue = np.array([120, 255, 255])
        lower_green = np.array([40, 140, 180])
        upper_green = np.array([70, 255, 255])

        # Create masks
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get center points
        def get_centers(contours):
            points = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
            return np.array(points)

        return {
            "red": get_centers(contours_red),
            "blue": get_centers(contours_blue),
            "green": get_centers(contours_green)
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"red": np.array([]), "blue": np.array([]), "green": np.array([])}

def show_pixel_info(event, x, y, flags, param):
    """
    Displays pixel information (BGR and HSV values) in the top-left corner of the image.
    """
    img, img_display = param
    if event == cv2.EVENT_MOUSEMOVE:
        img_display[:] = img.copy()

        # Get BGR values
        b, g, r = img[y, x]
        hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv_color

        # Text content
        text1 = f"X: {x}, Y: {y}"
        text2 = f"BGR: ({b}, {g}, {r})"
        text3 = f"HSV: ({h}, {s}, {v})"

        # Display text in the top-left corner
        cv2.putText(img_display, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_display, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_display, text3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Detected Points", img_display)

if __name__ == "__main__":
    image_file = "XArm6_Muñoz/After.png"

    # Detect points
    colored_points = detect_colored_points(image_file)  
    print("Red points:\n", colored_points["red"])
    print("Blue points:\n", colored_points["blue"])
    print("Green points:\n", colored_points["green"])

    # Load the image
    img = cv2.imread(image_file)
    if img is None:
        print(f"Error: Could not load image {image_file}")
        exit()

    img_display = img.copy()

    # Draw detected points
    for x, y in colored_points["red"]:
        cv2.circle(img_display, (x, y), 5, (255, 255, 255), -1)  # White
    for x, y in colored_points["blue"]:
        cv2.circle(img_display, (x, y), 5, (0, 0, 0), -1)  # Black
    for x, y in colored_points["green"]:
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)  # Green

    clicked_points = []

    def mouse_handler(event, x, y, flags, param):
        img, original_img = param

        # Always start from a clean image
        img_display[:] = original_img.copy()

        # Redraw persistent points
        for x_r, y_r in colored_points["red"]:
            cv2.circle(img_display, (x_r, y_r), 5, (255, 255, 255), -1)
        for x_b, y_b in colored_points["blue"]:
            cv2.circle(img_display, (x_b, y_b), 5, (0, 0, 0), -1)
        for x_g, y_g in colored_points["green"]:
            cv2.circle(img_display, (x_g, y_g), 5, (0, 255, 0), -1)

        # Redraw clicked points and lines
        for i, pt in enumerate(clicked_points):
            cv2.circle(img_display, pt, 4, (0, 255, 255), -1)
            if i >= 1:
                pt1 = clicked_points[i - 1]
                pt2 = clicked_points[i]
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                mid_x = (pt1[0] + pt2[0]) // 2
                mid_y = (pt1[1] + pt2[1]) // 2
                text = f"dx: {dx}, dy: {dy}"
                cv2.line(img_display, pt1, pt2, (0, 255, 255), 2)
                cv2.putText(img_display, text, (mid_x + 5, mid_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Show pixel info on hover
        if event == cv2.EVENT_MOUSEMOVE:
            b, g, r = img[y, x]
            hsv_color = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = hsv_color
            text1 = f"X: {x}, Y: {y}"
            text2 = f"BGR: ({b}, {g}, {r})"
            text3 = f"HSV: ({h}, {s}, {v})"
            cv2.putText(img_display, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img_display, text2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img_display, text3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Handle clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))

        cv2.imshow("Detected Points", img_display)


    # Set combined mouse callback
    cv2.imshow("Detected Points", img_display)
    cv2.setMouseCallback("Detected Points", mouse_handler, (img_display, img))

    print("\nHover to inspect pixels. Click to draw lines. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
