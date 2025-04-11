import os
import subprocess
import shutil
import sys
import open3d as o3d

# === Check if COLMAP is available ===
colmap_path = shutil.which("colmap")
if colmap_path is None:
    print("❌ COLMAP is not found in your system PATH. Make sure it's installed and added to PATH.")
    sys.exit(1)
print(f"✅ COLMAP found at: {colmap_path}")

# === Check if OpenMVS tools are available ===
def require_tool(tool):
    if shutil.which(tool) is None:
        print(f"❌ {tool} is not found in your system PATH. Install OpenMVS and add it to PATH.")
        sys.exit(1)
    else:
        print(f"✅ {tool} found.")

for tool in ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh"]:
    require_tool(tool)

# === Paths ===
image_dir = "data/data4"
workspace_dir = "colmap_out"
database_path = os.path.join(workspace_dir, "database.db")
sparse_dir = os.path.join(workspace_dir, "sparse")
mvs_dir = os.path.join(workspace_dir, "mvs_model")
dense_dir = os.path.join(workspace_dir, "dense")

output_model = os.path.join(workspace_dir, "boxboxbox.ply")
scene_path = os.path.join(workspace_dir, "scene.mvs")

# === Clean up if needed ===
if os.path.exists(database_path):
    os.remove(database_path)

os.makedirs(workspace_dir, exist_ok=True)
os.makedirs(sparse_dir, exist_ok=True)
os.makedirs(mvs_dir, exist_ok=True)

# === Step 1: Feature Extraction ===
print("📸 Extracting features...")
subprocess.run([
    colmap_path, "feature_extractor",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--ImageReader.single_camera", "1",
    "--ImageReader.camera_model", "PINHOLE",  # 👈 AÑADE ESTO
    "--SiftExtraction.estimate_affine_shape", "1"
], check=True)

# === Step 2: Feature Matching ===
print("🔗 Matching features...")
subprocess.run([
    colmap_path, "exhaustive_matcher",
    "--database_path", database_path
], check=True)

# === Step 3: Sparse Reconstruction (Mapping) ===
print("🧠 Running mapper...")
subprocess.run([
    colmap_path, "mapper",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--output_path", sparse_dir
], check=True)

# === Ensure sparse/0 exists ===
sparse_model_path = os.path.join(sparse_dir, "0")
if not os.path.isdir(sparse_model_path):
    print(f"❌ Expected sparse model at {sparse_model_path}, but it was not created.")
    sys.exit(1)

# === Step 4: Export model as TXT (for OpenMVS) ===
print("🔄 Exporting COLMAP model to TXT format...")
subprocess.run([
    colmap_path, "model_converter",
    "--input_path", sparse_model_path,
    "--output_path", mvs_dir,
    "--output_type", "TXT"
], check=True)

# === Check for required TXT files ===
required_txts = ["cameras.txt", "images.txt", "points3D.txt"]
for fname in required_txts:
    fpath = os.path.join(mvs_dir, fname)
    if not os.path.exists(fpath):
        print(f"❌ Missing required COLMAP TXT file: {fpath}")
        sys.exit(1)

# === Step 5: Convert to OpenMVS format ===
print("🔄 Converting to OpenMVS format...")
subprocess.run([
    "InterfaceCOLMAP",
    "-i", mvs_dir,
    "-o", scene_path,
    "-w", image_dir
], check=True)

# === Step 6: Densify the point cloud ===
print("🔁 Densifying with OpenMVS...")
subprocess.run([
    "DensifyPointCloud", scene_path, "--resolution-level", "1"
], check=True)

# === Step 7: Reconstruct the mesh ===
print("🔺 Reconstructing mesh...")
dense_mvs_path = scene_path.replace(".mvs", "_dense.mvs")
subprocess.run([
    "ReconstructMesh", dense_mvs_path
], check=True)

# === Step 8: (Optional) Texture the mesh ===
print("🖼️ Texturing the mesh...")
mesh_mvs_path = dense_mvs_path.replace(".mvs", "_mesh.mvs")
subprocess.run([
    "TextureMesh", mesh_mvs_path
], check=True)

# === Step 9: Load and visualize mesh in Open3D ===
print("🎉 Loading final textured mesh...")
textured_mesh_ply = mesh_mvs_path.replace(".mvs", ".ply")
mesh = o3d.io.read_triangle_mesh(textured_mesh_ply)
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(os.path.join(workspace_dir, "mesh_textured.ply"), mesh)
o3d.visualization.draw_geometries([mesh])
