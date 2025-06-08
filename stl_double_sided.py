import trimesh
import numpy as np

def make_stl_double_sided(input_path, output_path):
    # Load the STL mesh
    mesh = trimesh.load(input_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input file is not a valid triangle mesh.")

    # Duplicate the faces with reversed winding
    flipped_faces = mesh.faces[:, [0, 2, 1]]
    
    # Concatenate original and flipped
    double_faces = np.vstack((mesh.faces, flipped_faces))
    
    # Duplicate the corresponding vertices
    # Note: trimesh handles shared vertices internally, so we just reuse the same vertices
    double_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=double_faces, process=False)

    # Save the new mesh
    double_mesh.export(output_path)
    print(f"✅ Double-sided mesh saved to {output_path}")

# Example usage
input_stl = 'OilPanBasic.stl'
output_stl = 'OilPanBasic_DoubleSided.stl'

make_stl_double_sided(input_stl, output_stl)
