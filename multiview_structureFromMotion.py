import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import open3d as o3d

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
    orb = cv2.ORB_create(nfeatures=3000)  # feel free to increase
    for img in images:
        kps, des = orb.detectAndCompute(img, None)
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
    flann = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
except Exception as e:
    print("❌ Error initializing matcher:", e)
    exit(1)

camera_poses = [np.hstack((np.eye(3), np.zeros((3,1))))]
points3D = []

# === Bootstrap with first two images ===
try:
    if des_all[0] is None or des_all[1] is None:
        raise ValueError("One of the first two descriptors is None.")

    # des0 = np.asarray(des_all[0], dtype=np.float32)
    # des1 = np.asarray(des_all[1], dtype=np.float32)
    des0 = des_all[0]
    des1 = des_all[1]


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
    
    print("test1")
    P1 = K @ camera_poses[0]
    P2 = K @ camera_poses[1]
    
    print("🔍 Triangulation inputs:")
    print(f"P1 shape: {P1.shape}, P2 shape: {P2.shape}")
    print(f"pts1 shape: {pts1.T.shape}, pts2 shape: {pts2.T.shape}")
    print(f"pts1[:5]: {pts1[:5]}")
    print(f"pts2[:5]: {pts2[:5]}")
    print("P1:", P1)
    print("P2:", P2)
    print("pts1.T dtype:", pts1.T.dtype, "shape:", pts1.T.shape)
    print("pts2.T dtype:", pts2.T.dtype, "shape:", pts2.T.shape)
    print("pts1.T:", pts1.T[:,:5])  # show just first 5
    print("pts2.T:", pts2.T[:,:5])

    pts1f = np.array(pts1.T, dtype=np.float32)
    pts2f = np.array(pts2.T, dtype=np.float32)

    try:
        pts4D = cv2.triangulatePoints(P1, P2, pts1f, pts2f).astype(np.float64)
        print("✅ Triangulation succeeded")
    except Exception as tri_e:
        print("❌ Error in cv2.triangulatePoints:", tri_e)
        exit(1)

    print("test3")
    valid = np.abs(pts4D[3]) > 1e-6
    pts4D = pts4D[:, valid]
    print("test4")
    if pts4D.shape[1] == 0:
        raise ValueError("No valid 3D points after triangulation.")
    pts4D /= pts4D[3]
    points3D = pts4D[:3].T
    
    descriptors3D = [des_all[0][m.queryIdx] for m, n in matches
                     if m.distance < 0.8 * n.distance and m.queryIdx < len(des_all[0])]
    # descriptors3D = np.asarray(descriptors3D, dtype=np.float32)
    descriptors3D = np.asarray(descriptors3D)

except Exception as e:
    print("❌ Error during bootstrap:", repr(e))
    exit(1)

# === Add new views one by one ===
for i in tqdm(range(2, len(images))):
    try:
        if des_all[i] is None or des_all[i-1] is None:
            print(f"⚠️ Skipping view {i} due to missing descriptors.")
            continue

        kps = kps_all[i]
        des = des_all[i]

        # matches = flann.knnMatch(descriptors3D.astype(np.float32), des, k=2)
        matches = flann.knnMatch(descriptors3D, des, k=2)

        pts3D = []
        pts2D = []
        for j, (m, n) in enumerate(matches):
            if m.queryIdx < len(points3D) and m.distance < 0.8 * n.distance:
                pts3D.append(points3D[m.queryIdx])
                pts2D.append(kps[m.trainIdx].pt)

        print(f"View {i}: matched {len(matches)} keypoints")

        if len(pts3D) < 6:
            print(f"⚠️ Not enough matches for view {i}, skipping.")
            continue

        pts3D = np.array(pts3D).astype(np.float32)
        pts2D = np.array(pts2D).astype(np.float32)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(pts3D, pts2D, K, dist)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3,1)
        camera_poses.append(np.hstack((R, t)))

        if i >= len(camera_poses):
            print(f"⚠️ Missing camera pose for view {i}, skipping triangulation.")
            continue

        matches = flann.knnMatch(des_all[i-1], des_all[i], k=2)
        pts_prev = []
        pts_curr = []
        new_desc = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                pts_prev.append(kps_all[i-1][m.queryIdx].pt)
                pts_curr.append(kps_all[i][m.trainIdx].pt)
                new_desc.append(des_all[i-1][m.queryIdx])

        if len(pts_prev) >= 8 and len(pts_prev) == len(pts_curr):
            P1 = K @ camera_poses[i-1]
            P2 = K @ camera_poses[i]
            pts4D = cv2.triangulatePoints(P1, P2, np.array(pts_prev).T, np.array(pts_curr).T)
            pts4D = pts4D.astype(np.float64)
            valid = np.abs(pts4D[3]) > 1e-6
            pts4D = pts4D[:, valid]
            if pts4D.shape[1] == 0:
                print(f"⚠️ No valid triangulated points at view {i}, skipping.")
                continue
            pts4D /= pts4D[3]
            new_pts = pts4D[:3].T
            print(f"View {i}: {len(new_pts)} points triangulated")
            points3D = np.vstack((points3D, new_pts))
            if new_desc:
                descriptors3D = np.vstack((descriptors3D, np.array(new_desc)))
    except Exception as e:
        print(f"❌ Exception at view {i}:", e)
        continue

# === Save final point cloud with color, denoising and normalization ===
try:
    points3D_filtered = points3D[np.isfinite(points3D).all(axis=1)]
    np.save("full_multiview_points.npy", points3D_filtered)

    if points3D_filtered.shape[0] == 0:
        print("❌ No valid 3D points to save.")
        exit(1)

    print("✨ Coloring and saving point cloud...")

    # Use the first image for color projection
    img_color = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)  # or IMREAD_COLOR for color
    height, width = img_color.shape[:2]

    colors = []
    for pt in points3D_filtered:
        pt_hom = np.append(pt, 1.0)
        uv = K @ camera_poses[0] @ pt_hom
        uv /= uv[2]
        u, v = int(round(uv[0])), int(round(uv[1]))
        if 0 <= u < width and 0 <= v < height:
            b, g, r = img_color[v, u]
            colors.append([r / 255.0, g / 255.0, b / 255.0])  # Open3D expects RGB
        else:
            colors.append([0, 0, 0])  # black for out-of-bounds

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D_filtered)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # === Denoising ===
    print("🧽 Removing statistical outliers...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # === Normalize ===
    print("📐 Centering and normalizing...")
    pts = np.asarray(pcd.points)
    pts -= pts.mean(axis=0)
    pts /= np.linalg.norm(pts, axis=1).max()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # === Save ===
    o3d.io.write_point_cloud("full_multiview_cloud.ply", pcd)
    print("✅ Saved colored, denoised, normalized 3D point cloud as full_multiview_cloud.ply")

except Exception as e:
    print("❌ Error saving point cloud:", e)
    exit(1)
