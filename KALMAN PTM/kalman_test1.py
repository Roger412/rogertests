import numpy as np
import matplotlib.pyplot as plt

# Time step
dt = 1.0

# Define the system
F = np.array([[1, dt],
              [0, 1]])  # State transition matrix
H = np.array([[1, 0]])  # Observation matrix
Q = np.array([[1, 0],
              [0, 3]])  # Process noise covariance
R = np.array([[10]])    # Measurement noise covariance
P = np.array([[500, 0],
              [0, 500]])  # Initial estimate error covariance
x = np.array([[0],
              [0]])  # Initial state (position and velocity)

# Simulated measurements (e.g., position measurements over time)
measurements = [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11,23,3,4,6,3,21,3,7,8,2,12,1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 1, 3, 2, 5, 4, 7, 6, 9, 8, 11,23,3,4,6,3,21,3,7,8,2,12]

# Lists to store estimates
estimated_positions = []
estimated_velocities = []

for z in measurements:
    # Prediction
    x = F @ x
    P = F @ P @ F.T + Q

    # Measurement Update
    y = np.array([[z]]) - H @ x  # Measurement residual
    S = H @ P @ H.T + R          # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

    x = x + K @ y
    P = (np.eye(2) - K @ H) @ P

    # Store estimates
    estimated_positions.append(x[0, 0])
    estimated_velocities.append(x[1, 0])

# Plotting the results
plt.plot(measurements, label='Measurements', marker='o')
plt.plot(estimated_positions, label='Estimated Position')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.legend()
plt.title('Kalman Filter Position Estimation')
plt.show()
