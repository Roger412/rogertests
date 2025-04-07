import open3d as o3d
import numpy as np

# Load saved 3D points
points = np.load("triangulated_points.npy")

# Optional: remove outliers (if not already filtered)
points = points[(points[:,2] > 0) & (points[:,2] < 3000)]

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

print("Min XYZ:", np.min(points, axis=1))
print("Max XYZ:", np.max(points, axis=1))

print("Number of matches:", len(points))

points = points[(points[:,2] > 0) & (points[:,2] < 3000)]
points = points * 1000  # Optional: scale to visible size

# Save as PLY file
o3d.io.write_point_cloud("triangulated_pointcloud.ply", pcd)
print("✅ Saved point cloud to triangulated_pointcloud.ply")
