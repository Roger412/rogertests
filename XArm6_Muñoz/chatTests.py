import numpy as np
import matplotlib.pyplot as plt

# Puntos originales en un cuadrado
square = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
    [0, 0]  # para cerrar el cuadrado
]).T  # forma 2xN

def apply_transform(matrix, points):
    """Aplica transformación homogénea a puntos 2D"""
    homog_points = np.vstack((points, np.ones((1, points.shape[1]))))  # 3xN
    transformed = matrix @ homog_points
    transformed /= transformed[2, :]  # normalizar homogéneas
    return transformed[:2, :]

# Definimos transformaciones homogéneas 3x3
# 1. Isometría (rotación + traslación)
theta = np.pi / 6
R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
t = np.array([[1], [0.5]])
T_iso = np.block([
    [R, t],
    [np.zeros((1, 2)), 1]
])

# 2. Similitud (escala uniforme + rotación + traslación)
s = 1.5
T_sim = np.block([
    [s * R, t],
    [np.zeros((1, 2)), 1]
])

# 3. Afín (shear por ejemplo)
A_affine = np.array([
    [1, 0.5],
    [0.2, 1]
])
T_aff = np.block([
    [A_affine, t],
    [np.zeros((1, 2)), 1]
])

# 4. Proyectiva (con perspectiva)
T_proj = np.array([
    [1, 0.2, 0],
    [0.2, 1, 0],
    [0.001, 0.001, 1]
])

# Aplicamos transformaciones
transformed = {
    "Original": square,
    "Isometría": apply_transform(T_iso, square),
    "Similitud": apply_transform(T_sim, square),
    "Afín": apply_transform(T_aff, square),
    "Proyectiva": apply_transform(T_proj, square)
}

# Plot
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
for i, (name, pts) in enumerate(transformed.items()):
    ax[i].plot(*pts, marker='o')
    ax[i].set_title(name)
    ax[i].axis('equal')
    ax[i].grid(True)

plt.tight_layout()
plt.show()
