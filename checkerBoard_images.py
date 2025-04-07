import cv2
import numpy as np

# Size of internal corners (not squares). Adjust if your pattern is different.
CHECKERBOARD = (11, 5)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Couldn't open camera.")
    exit()

i = 0  # image index

print("📷 Press 's' to save a frame with detected corners.")
print("❌ Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Make a copy BEFORE drawing (used for saving)
    raw_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try to find the checkerboard corners
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if found:
        # Refine corner positions (for subpixel accuracy)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        # Draw the corners (for display only)
        cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)
        cv2.putText(frame, "✅ Corners found - Press 's' to save", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    else:
        cv2.putText(frame, "❌ No checkerboard detected", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Checkerboard Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and found:
        fname = f"calib_imgs/calib_live_{i}.png"
        cv2.imwrite(fname, raw_frame)  # ✅ Save the raw (undrawn) frame
        print(f"💾 Saved clean image: {fname}")
        i += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
