import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm

# === Load calibration data ===
try:
    calib_data = np.load("calibration_data.npz")
    K = calib_data["K"]
    dist = calib_data["dist"]
except Exception as e:
    print("❌ Error loading calibration data:", e)
    exit(1)

# === Load all images ===
try:
    image_paths = sorted(glob("data/capture_*.png"))
    images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    assert all(img is not None for img in images), "Failed to load all images"
except Exception as e:
    print("❌ Error loading images:", e)
    exit(1)

# === Detect SIFT features in all images ===
kps_all = []
des_all = []
try:
    sift = cv2.SIFT_create()
    for img in images:
        kps, des = sift.detectAndCompute(img, None)
        kps_all.append(kps)
        des_all.append(des)
except Exception as e:
    print("❌ Error in feature detection:", e)
    exit(1)

# === Initialize matching and 3D reconstruction ===
try:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
except Exception as e:
    print("❌ Error initializing matcher:", e)
    exit(1)

camera_poses = [np.hstack((np.eye(3), np.zeros((3,1))))]
points3D = []

# === Bootstrap with first two images ===
try:
    if des_all[0] is None or des_all[1] is None:
        raise ValueError("One of the first two descriptors is None.")

    des0 = np.asarray(des_all[0], dtype=np.float32)
    des1 = np.asarray(des_all[1], dtype=np.float32)

    print("des0 shape:", des0.shape, des0.dtype)
    print("des1 shape:", des1.shape, des1.dtype)
    print("kps0:", len(kps_all[0]))
    print("kps1:", len(kps_all[1]))

    try:
        matches = flann.knnMatch(des0, des1, k=2)
    except cv2.error as e:
        print("❌ OpenCV FLANN error:", e)
        exit(1)

    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            pts1.append(kps_all[0][m.queryIdx].pt)
            pts2.append(kps_all[1][m.trainIdx].pt)

    if len(pts1) < 8:
        raise ValueError(f"Not enough good matches for initialization: {len(pts1)} found.")

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        raise ValueError("Essential matrix computation failed.")

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    camera_poses.append(np.hstack((R, t)))

    P1 = K @ camera_poses[0]
    P2 = K @ camera_poses[1]
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T).astype(np.float64)
    valid = np.abs(pts4D[3]) > 1e-6
    pts4D = pts4D[:, valid]
    if pts4D.shape[1] == 0:
        raise ValueError("No valid 3D points after triangulation.")
    pts4D /= pts4D[3]
    points3D = pts4D[:3].T

    descriptors3D = [des_all[0][m.queryIdx] for m, n in matches
                     if m.distance < 0.8 * n.distance and m.queryIdx < len(des_all[0])]
    descriptors3D = np.asarray(descriptors3D, dtype=np.float32)

except Exception as e:
    print("❌ Error during bootstrap:", repr(e))
    exit(1)

