import cv2
import numpy as np
import pandas as pd

img1_path = "data/capture_000.png"
img2_path = "data/capture_001.png"

# === Load calibration data ===
calib_data = np.load("calibration_data.npz")
K = calib_data["K"]
dist = calib_data["dist"]

# === Load stereo images ===
img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# === Detect SIFT features ===
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# === Match descriptors using FLANN + Lowe's ratio test ===
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# === Estimate essential matrix ===
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# === Recover relative camera pose (R, t) ===
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# === Compute projection matrices ===
P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K @ np.hstack((R, t))

# === Triangulate matching points ===
pts4D_hom = cv2.triangulatePoints(P1, P2, pts1.T.astype(float), pts2.T.astype(float))
pts3D = pts4D_hom[:3] / pts4D_hom[3]

# === Filter and format as DataFrame for display/export ===
df = pd.DataFrame(pts3D.T, columns=["X", "Y", "Z"])
df = df[(df["Z"] > 0) & (df["Z"] < 3000)]

# Save filtered 3D points to a .npy file for Open3D
np.save("triangulated_points.npy", df[["X", "Y", "Z"]].values)
print("✅ Saved 3D points to triangulated_points.npy")

