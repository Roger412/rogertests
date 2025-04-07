import cv2
import numpy as np
import glob

# === USER PARAMETERS ===
CHECKERBOARD = (11, 5)      # corners, NOT squares
SQUARE_SIZE = 10           # in mm (or the units you measured)
IMAGE_GLOB = "calib_imgs/calib_*.png" # your image pattern

# === PREPARE OBJECT POINTS ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE  # Scale to real world

objpoints = []  # 3D points in world
imgpoints = []  # 2D points in image

images = glob.glob(IMAGE_GLOB)

print(f"🧮 Found {len(images)} images to process.")


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # Refine corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Optional: show detected corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

    if ret:
        print(f"✅ Checkerboard found in: {fname}")
        ...
    else:
        print(f"❌ No checkerboard found in: {fname}")

cv2.destroyAllWindows()

# === CALIBRATE ===
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# === PRINT RESULTS ===
print("✅ Calibration successful!" if ret else "❌ Calibration failed.")
print("\n🎯 Camera Intrinsic Matrix (K):\n", K)
print("\n📏 Distortion Coefficients:\n", dist.ravel())
print("\n🌀 Reprojection Error:", ret)


img = cv2.imread("calib_imgs/calib_live_0.png")

h, w = img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
undistorted = cv2.undistort(img, K, dist, None, new_K)

cv2.imshow("Undistorted", undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()

np.savez("calibration_data.npz", K=K, dist=dist)
