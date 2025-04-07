
# === Add new views one by one ===
for i in tqdm(range(2, len(images))):
    try:
        if des_all[i] is None or des_all[i-1] is None:
            print(f"⚠️ Skipping view {i} due to missing descriptors.")
            continue

        kps = kps_all[i]
        des = des_all[i]

        matches = flann.knnMatch(descriptors3D.astype(np.float32), des, k=2)
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

# === Save final point cloud ===
try:
    points3D_filtered = points3D[np.isfinite(points3D).all(axis=1)]
    np.save("full_multiview_points.npy", points3D_filtered)
    if points3D_filtered.shape[0] == 0:
        print("❌ No valid 3D points to save.")
        exit(1)

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D_filtered)
    o3d.io.write_point_cloud("full_multiview_cloud.ply", pcd)
    print("✅ Saved full 3D point cloud as full_multiview_cloud.ply")
except Exception as e:
    print("❌ Error saving point cloud:", e)
    exit(1)