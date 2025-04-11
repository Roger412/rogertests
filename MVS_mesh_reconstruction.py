import os
import subprocess
import shutil
import sys
import open3d as o3d

# === Check if OpenMVS tools are available ===
def require_tool(tool):
    if shutil.which(tool) is None:
        print(f"❌ {tool} is not found in your system PATH. Install OpenMVS and add it to PATH.")
        sys.exit(1)
    else:
        print(f"✅ {tool} found.")

for tool in ["InterfaceCOLMAP", "DensifyPointCloud", "ReconstructMesh", "TextureMesh"]:
    require_tool(tool)


# === Paso 5: Convertir a formato OpenMVS ===
os.system("InterfaceCOLMAP -i MVS_ws -o MVS_ws/mvs_files/mvs_scene.mvs --image-folder images")

# === Paso 6: Densificar ===
os.system("DensifyPointCloud MVS_ws/mvs_files/mvs_scene.mvs --resolution-level 1")

# === Paso 7: Reconstrucción de malla ===
os.system("ReconstructMesh MVS_ws/mvs_files/mvs_scene_dense.mvs -p MVS_ws/mvs_files/mvs_scene_dense.ply")

os.system("RefineMesh MVS_ws/mvs_files/mvs_scene.mvs -m MVS_ws/mvs_files/mvs_scene_dense_mesh.ply -o mvs_scene_dense_mesh_refined.mvs")

os.system("TextureMesh MVS_ws/mvs_files/mvs_scene_dense.mvs -m MVS_ws/mvs_files/mvs_scene_dense_mesh_refined.ply -o MVS_ws/mvs_files/mvs_scene_dense_mesh_refined_texture.mvs")

