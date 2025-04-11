import cv2
import numpy as np
import open3d as o3d  # Asegúrate de tenerlo: pip install open3d
import matplotlib.pyplot as plt

# === 1. CARGAR IMÁGENES ===
img1 = cv2.imread('data/cabaña2/cabaña01.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('data/cabaña2/cabaña02.jpeg', cv2.IMREAD_GRAYSCALE)

# === 2. DETECCIÓN ORB Y MATCHING ===
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

# Filtrar puntos
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# === 3. MATRIZ FUNDAMENTAL + MÁSCARA DE INLIERS ===
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# === 4. MATRIZ ESENCIAL (Asumimos cámara calibrada) ===
K = np.array([[1000, 0, img1.shape[1]//2],
              [0, 1000, img1.shape[0]//2],
              [0,    0,              1]])

E = K.T @ F @ K
_, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

# === 5. MATRICES DE PROYECCIÓN ===
P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K @ np.hstack((R, t))

# === 6. TRIANGULACIÓN ===
pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, None)
pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, None)

points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
points_3d = (points_4d / points_4d[3])[:3].T

# === 7. EXPORTAR COMO .PLY ===
def save_point_cloud_ply(filename, points, colors=None):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, point_cloud)

# Colores desde imagen 1 (solo para visual bonito)
colors = []
for pt in pts1.astype(int):
    y, x = pt[1], pt[0]
    color = img1[int(y), int(x)] / 255.0
    colors.append([color]*3)  # gris

save_point_cloud_ply("cabaña_3d.ply", points_3d, colors)

print("✅ Nube de puntos guardada en 'cabaña_3d.ply'. Ábrela con MeshLab, CloudCompare o Open3D.")

# === 8. OPCIONAL: Visualizar con Open3D ===
pcd = o3d.io.read_point_cloud("cabaña_3d.ply")
o3d.visualization.draw_geometries([pcd])
