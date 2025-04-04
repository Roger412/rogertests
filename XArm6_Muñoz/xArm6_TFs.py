import numpy as np
import sympy as sp

# Transformation Matrix for Joint1
joint1_transform = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 267],
    [0, 0, 0, 1]
])

# Define symbolic variables for Joint2 and Joint3 offsets
t2_offset = sp.symbols('t2_offset')
t3_offset = sp.symbols('t3_offset')
a2 = sp.symbols('a2')

# Transformation Matrix for Joint2
joint2_transform = np.array([
    [sp.cos(t2_offset), -sp.sin(t2_offset), 0, 0],
    [0, 0, 1, 0],
    [-sp.sin(t2_offset), -sp.cos(t2_offset), 0, 0],
    [0, 0, 0, 1]
])

# Transformation Matrix for Joint3
joint3_transform = np.array([
    [sp.cos(t3_offset), -sp.sin(t3_offset), 0, a2],
    [sp.sin(t3_offset), sp.cos(t3_offset), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Transformation Matrix for Joint4
joint4_transform = np.array([
    [1, 0, 0, 155/2],
    [0, 0, 1, 685/2],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

# Transformation Matrix for Joint5
joint5_transform = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

# Transformation Matrix for Joint6
joint6_transform = np.array([
    [1, 0, 0, 76],
    [0, 0, 1, 97],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

# Print the matrices (optional)
print("Transformation Matrix for Joint1:\n", joint1_transform)
print("\nTransformation Matrix for Joint2:\n", joint2_transform)
print("\nTransformation Matrix for Joint3:\n", joint3_transform)
print("Transformation Matrix for Joint4:\n", joint4_transform)
print("\nTransformation Matrix for Joint5:\n", joint5_transform)
print("\nTransformation Matrix for Joint6:\n", joint6_transform)
