import os
import subprocess
import shutil
import sys

# === Check if COLMAP is available ===
colmap_path = shutil.which("colmap")
if colmap_path is None:
    print("❌ COLMAP is not found in your system PATH. Make sure it's installed and added to PATH.")
    sys.exit(1)

print(f"✅ COLMAP found at: {colmap_path}")

# === Paths ===
project_dir = "colmap_test"
image_dir = "data/data1/"
database_path = os.path.join(project_dir, "database.db")
sparse_dir = os.path.join(project_dir, "sparse")
output_model = os.path.join(project_dir, "model.ply")

os.makedirs(project_dir, exist_ok=True)

# === Step 1: Feature Extraction ===
subprocess.run([
    colmap_path, "feature_extractor",
    "--database_path", database_path,
    "--image_path", image_dir
], check=True)

# === Step 2: Feature Matching ===
subprocess.run([
    colmap_path, "exhaustive_matcher",
    "--database_path", database_path
], check=True)

# === Step 3: Sparse Reconstruction ===
os.makedirs(sparse_dir, exist_ok=True)
subprocess.run([
    colmap_path, "mapper",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--output_path", sparse_dir
], check=True)

# === Step 4: Export model as PLY ===
subprocess.run([
    colmap_path, "model_converter",
    "--input_path", os.path.join(sparse_dir, "0"),
    "--output_path", output_model,
    "--output_type", "PLY"
], check=True)

print("✅ Done! 3D model saved to:", output_model)
