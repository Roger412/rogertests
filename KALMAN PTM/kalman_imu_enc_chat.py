import numpy as np

class EKFTricycleState:
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0, bias_imu=0.0, bias_gyro=0.0):
        self.x = np.array([
            x,          # x_position
            y,          # y_position
            theta,      # heading
            v,          # linear velocity
            omega,      # angular velocity
            bias_imu,   # accelerometer bias
            bias_gyro   # gyroscope bias
        ], dtype=float)

    def __getitem__(self, index):
        return self.x[index]

    def __setitem__(self, index, value):
        self.x[index] = value

    def as_vector(self):
        return self.x.copy()

    def predict_state(self, dt, u):
        """
        Applies the velocity-based motion model with control input:
        u = [v_cmd, omega_cmd]
        """
        x, y, theta, _, _, bias_imu, bias_gyro = self.x
        v_cmd, omega_cmd = u

        # Predict new pose
        x_new = x + v_cmd * np.cos(theta) * dt
        y_new = y + v_cmd * np.sin(theta) * dt
        theta_new = theta + omega_cmd * dt

        # Replace velocities with commanded ones
        self.x = np.array([
            x_new,
            y_new,
            theta_new,
            v_cmd,
            omega_cmd,
            bias_imu,
            bias_gyro
        ])

    def jacobian_A(self, dt):
        """Jacobian of the motion model ∂f/∂x, using current commanded velocities."""
        _, _, theta, _, _, _, _ = self.x
        v_cmd = self.x[3]

        A = np.eye(7)
        A[0, 2] = -v_cmd * np.sin(theta) * dt
        A[0, 3] =  np.cos(theta) * dt
        A[1, 2] =  v_cmd * np.cos(theta) * dt
        A[1, 3] =  np.sin(theta) * dt
        A[2, 4] =  dt

        return A

    def h(self, dv_approx):
        """
        Measurement function h(x), using externally computed forward acceleration.
        dv_approx = (v_cmd[k] - v_cmd[k-1]) / dt
        """
        _, _, _, v, omega, bias_imu, bias_gyro = self.x
        return np.array([
            v,                            # encoder velocity
            omega,                        # encoder yaw rate
            omega + bias_gyro,            # IMU gyroscope
            dv_approx + bias_imu          # IMU forward acceleration (supposedly, since dv is calculaded with v_cmd)
        ])

    def jacobian_H(self, dt):
        """
        Jacobian H = ∂h/∂x for the measurement model.
        Depends on dt for ∂a/∂v = 1/dt
        """
        H = np.zeros((4, 7))

        # ∂v/∂v
        H[0, 3] = 1.0

        # ∂omega/∂omega
        H[1, 4] = 1.0

        # ∂(omega + bias_gyro)/∂omega and ∂.../∂bias_gyro
        H[2, 4] = 1.0
        H[2, 6] = 1.0

        # ∂(dv + bias_imu)/∂v and ∂.../∂bias_imu
        H[3, 3] = 1.0 / dt  # ∂(dv) ≈ ∂(Δv)/∂v
        H[3, 5] = 1.0

        return H
