import open3d as o3d

# Load the textured mesh
mesh = o3d.io.read_triangle_mesh("MVS_ws/mvs_files/mvs_scene_dense_mesh_refined_texture.ply")

# Optional: enable smooth shading and normals
mesh.compute_vertex_normals()

# View it
o3d.visualization.draw_geometries([mesh])
