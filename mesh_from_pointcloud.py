
import numpy as np
import open3d as o3d
from skimage import measure
from scipy.spatial import cKDTree

pc_path = "colmap_out/dense/fused.ply"
pcd = o3d.io.read_point_cloud("colmap_out/dense/fused.ply")

# Convert Open3D point cloud to numpy array
points = np.asarray(pcd.points)

mins = np.min(points, axis=0)
maxs = np.max(points, axis=0)

voxel_size=0.5 
iso_level_percentile=20


# Create a 3D grid
x = np.arange(mins[0], maxs[0], voxel_size)
y = np.arange(mins[1], maxs[1], voxel_size)
z = np.arange(mins[2], maxs[2], voxel_size)
x, y, z = np.meshgrid(x, y, z, indexing='ij')

# Create a KD-tree for efficient nearest neighbor search
tree = cKDTree(points)

# Compute the scalar field (distance to nearest point)
grid_points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
distances, _ = tree.query(grid_points)
scalar_field = distances.reshape(x.shape)

# Determine iso-level based on percentile of distances
iso_level = np.percentile(distances, iso_level_percentile)

# Apply Marching Cubes
verts, faces, _, _ = measure.marching_cubes(scalar_field, level=iso_level)

# Scale and translate vertices back to original coordinate system
verts = verts * voxel_size + mins

pcd.estimate_normals()

radii = [0.05, 0.1, 0.2]  # probá con radios según el tamaño del objeto
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))


o3d.io.write_triangle_mesh("ball_pivot_mesh.ply", mesh)


# Compute vertex normals
mesh.compute_vertex_normals()

# Visualize the result
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

o3d.io.write_triangle_mesh("Mesh_attempts/output_mesh.ply", mesh)
