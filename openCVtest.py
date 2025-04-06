import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Function to calculate angle with corrected orientation
def calculate_angle(point1, point2):
    """Calculate the angle of a vector and correct its orientation."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = np.arctan2(-dy, dx) * 180 / np.pi  # Negate dy to correct orientation
    return angle


# Function to detect colored circles
def detect_colored_circles(frame):
    """Detects red, green, and yellow circles in the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        "Red": [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),   # Lower red range
            (np.array([170, 120, 70]), np.array([180, 255, 255])) # Upper red range
        ],
        "Green": [(np.array([40, 40, 40]), np.array([80, 255, 255]))],  # Green
        "Yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))]  # Yellow
    }

    detected_circles = []

    for color, ranges in color_ranges.items():
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)  # Initialize mask
        
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv, lower, upper)  # Combine masks if needed

        blurred = cv2.GaussianBlur(mask, (9, 9), 2)

        # Detect circles using Hough Transform
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=10, maxRadius=100)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                detected_circles.append((x, y, r, color))

    return detected_circles


# Start Video Capture
cap = cv2.VideoCapture(0)  # Use default webcam

# Initialize MediaPipe Hands
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Detect up to 2 hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get key points (wrist and index finger tip)
                wrist = hand_landmarks.landmark[0]  # Wrist
                index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip

                # Convert to pixel coordinates
                h, w, _ = frame.shape
                wrist_pos = (int(wrist.x * w), int(wrist.y * h))
                index_finger_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

                # Calculate the angle
                angle = calculate_angle(wrist_pos, index_finger_pos)

                # Ensure the upward hand has 90° instead of -90°
                if angle < 0:
                    angle += 360

                # Display angle on screen
                cv2.putText(frame, f"Angle: {angle:.2f} deg", wrist_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Detect circles
        circles = detect_colored_circles(frame)

        # Draw detected circles
        for (x, y, r, color) in circles:
            if color == "Red":
                circle_color = (0, 0, 255)
            elif color == "Green":
                circle_color = (0, 255, 0)
            else:  # Yellow
                circle_color = (0, 255, 255)

            cv2.circle(frame, (x, y), r, circle_color, 3)
            cv2.putText(frame, color, (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle_color, 2)

        # Show the frame
        cv2.imshow("Hand Detection & Circle Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
