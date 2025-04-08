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
image_dir = "data/data4"
workspace_dir = "colmap_out"
database_path = os.path.join(workspace_dir, "database.db")
sparse_dir = os.path.join(workspace_dir, "sparse")
output_model = os.path.join(workspace_dir, "patotest.ply")

if os.path.exists(database_path):
    os.remove(database_path)

os.makedirs(workspace_dir, exist_ok=True)
os.makedirs(sparse_dir, exist_ok=True)

# === Step 1: Feature Extraction ===
print("📸 Extracting features...")
subprocess.run([
    colmap_path, "feature_extractor",
    "--database_path", database_path,
    "--image_path", image_dir,
    "--ImageReader.single_camera", "1",  # Helps when all images come from one camera
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

# === Step 4: Convert model to PLY ===
print("📦 Converting model to PLY...")
subprocess.run([
    colmap_path, "model_converter",
    "--input_path", os.path.join(sparse_dir, "0"),
    "--output_path", output_model,
    "--output_type", "PLY"
], check=True)

print("✅ Reconstruction finished! Model saved to:", output_model)
