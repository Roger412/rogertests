#!/usr/bin/env python3
"""
Differential Drive Robot EKF Sensor Fusion Implementation
Implements an 11-state EKF for 2D pose estimation using wheel encoders and 9DoF IMU
State: [x, y, θ, v_x, v_y, ω_z, b_gx, b_gy, b_gz, b_ax, b_ay]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List
from collections import deque

plt.style.use('seaborn-v0_8-darkgrid')

@dataclass
class SensorConfig:
    """Configuration for sensor simulation"""
    # Update rates
    wheel_encoder_hz: float = 50.0    # Wheel encoder update rate
    imu_gyro_hz: float = 100.0        # IMU gyroscope update rate  
    imu_accel_hz: float = 100.0       # IMU accelerometer update rate
    imu_orientation_hz: float = 20.0  # IMU orientation (magnetometer fusion) update rate
    
    # Noise characteristics
    wheel_velocity_noise_std: float = 0.05    # Linear velocity noise (m/s)
    wheel_angular_noise_std: float = 0.02     # Angular velocity noise (rad/s)
    gyro_noise_std: float = 0.01              # Gyroscope noise (rad/s)
    accel_noise_std: float = 0.1              # Accelerometer noise (m/s^2)
    orientation_noise_std: float = 0.05       # Orientation noise (rad)
    
    # Bias characteristics  
    gyro_bias_walk_std: float = 0.001         # Gyro bias random walk (rad/s per sqrt(s))
    accel_bias_walk_std: float = 0.01         # Accel bias random walk (m/s^2 per sqrt(s))
    initial_gyro_bias_std: float = 0.05       # Initial gyro bias (rad/s)
    initial_accel_bias_std: float = 0.2       # Initial accel bias (m/s^2)

@dataclass  
class EKFConfig:
    """Configuration for Extended Kalman Filter"""
    # Process noise
    pos_process_std: float = 0.01         # Position process noise
    vel_process_std: float = 0.1          # Velocity process noise  
    orientation_process_std: float = 0.05 # Orientation process noise
    gyro_bias_process_std: float = 0.001  # Gyro bias process noise
    accel_bias_process_std: float = 0.01  # Accel bias process noise
    
    # Initial covariance
    pos_init_std: float = 1.0             # Initial position uncertainty
    vel_init_std: float = 0.5             # Initial velocity uncertainty
    orientation_init_std: float = 0.1     # Initial orientation uncertainty
    gyro_bias_init_std: float = 0.1       # Initial gyro bias uncertainty
    accel_bias_init_std: float = 0.5      # Initial accel bias uncertainty
    
    # Initial state errors (for testing)
    init_pos_error: np.ndarray = None     # Will be set to np.array([0.5, -0.3])
    init_orientation_error: float = 0.2   # Initial orientation error (rad)

class RobotSimulator:
    """Simulates true differential drive robot motion"""
    
    def __init__(self, pattern='circle', duration=30.0, dt=0.01):
        self.pattern = pattern
        self.duration = duration
        self.dt = dt
        self.t = 0.0
        
        # Robot physical parameters
        self.wheel_radius = 0.05      # 5cm wheel radius
        self.wheelbase = 0.3          # 30cm wheelbase
        
        # State variables
        self.position = np.zeros(2)   # [x, y]
        self.orientation = 0.0        # θ (yaw angle)
        self.velocity = np.zeros(2)   # [v_x, v_y] in world frame
        self.angular_velocity = 0.0   # ω_z
        
        # Wheel speeds (for encoder simulation)
        self.wheel_speed_left = 0.0   # rad/s
        self.wheel_speed_right = 0.0  # rad/s
        
        # IMU measurements (body frame)
        self.angular_velocity_body = np.zeros(3)  # [ω_x, ω_y, ω_z]
        self.acceleration_body = np.zeros(3)      # [a_x, a_y, a_z] including gravity
        
    def step(self):
        """Update robot state based on trajectory pattern"""
        self.t += self.dt
        
        if self.pattern == 'straight':
            # Straight line motion
            v_desired = 1.0 if self.t < 10 else 0.5
            omega_desired = 0.0
            
        elif self.pattern == 'circle':
            # Circular trajectory
            v_desired = 0.8
            omega_desired = 0.3
            
        elif self.pattern == 'figure8':
            # Figure-8 pattern
            v_desired = 0.6
            omega_desired = 0.4 * np.sin(0.2 * self.t)
            
        elif self.pattern == 'square':
            # Square trajectory
            segment_time = 5.0
            segment = int(self.t / segment_time) % 4
            time_in_segment = self.t % segment_time
            
            if time_in_segment < 4.0:  # Move forward
                v_desired = 0.5
                omega_desired = 0.0
            else:  # Turn 90 degrees
                v_desired = 0.0
                omega_desired = np.pi/2  # 90 deg/s
                
        elif self.pattern == 'stop':
            v_desired = 0.0
            omega_desired = 0.0
            
        # Convert desired velocities to wheel speeds
        self.wheel_speed_left = (v_desired - omega_desired * self.wheelbase / 2) / self.wheel_radius
        self.wheel_speed_right = (v_desired + omega_desired * self.wheelbase / 2) / self.wheel_radius
        
        # Forward kinematics: wheel speeds -> robot motion
        v_robot = (self.wheel_speed_left + self.wheel_speed_right) * self.wheel_radius / 2
        omega_robot = (self.wheel_speed_right - self.wheel_speed_left) * self.wheel_radius / self.wheelbase
        
        # Update robot state
        self.angular_velocity = omega_robot
        self.orientation += omega_robot * self.dt
        
        # Keep orientation in [-π, π]
        self.orientation = np.arctan2(np.sin(self.orientation), np.cos(self.orientation))
        
        # Velocity in world frame
        self.velocity[0] = v_robot * np.cos(self.orientation)
        self.velocity[1] = v_robot * np.sin(self.orientation)
        
        # Update position
        self.position += self.velocity * self.dt
        
        # Simulate IMU measurements in body frame
        # Angular velocity (assuming robot moves on flat ground, so ωx ≈ ωy ≈ 0)
        self.angular_velocity_body = np.array([0.0, 0.0, omega_robot])
        
        # Acceleration in body frame (mainly gravity + small dynamics)
        # For simplicity, assume minimal linear acceleration except gravity
        accel_forward = 0.0  # Could add some dynamics here
        accel_lateral = 0.0
        
        # Gravity in body frame (robot tilts slightly during turns)
        roll_angle = 0.05 * omega_robot  # Small roll during turns
        pitch_angle = 0.02 * accel_forward  # Small pitch during acceleration
        
        self.acceleration_body = np.array([
            -9.81 * np.sin(pitch_angle) + accel_forward,
            -9.81 * np.sin(roll_angle) + accel_lateral,
            9.81 * np.cos(roll_angle) * np.cos(pitch_angle)
        ])
        
        return self.t < self.duration

class SensorSimulator:
    """Simulates noisy sensor measurements with configurable update rates"""
    
    def __init__(self, config: SensorConfig):
        self.config = config
        
        # Last update times
        self.last_wheel_time = 0.0
        self.last_gyro_time = 0.0
        self.last_accel_time = 0.0
        self.last_orientation_time = 0.0
        
        # Sensor biases (random walk)
        self.gyro_bias = np.random.normal(0, config.initial_gyro_bias_std, 3)
        self.accel_bias = np.random.normal(0, config.initial_accel_bias_std, 3)
        
    def update(self, robot: RobotSimulator, current_time: float) -> Dict:
        """Generate sensor measurements based on update rates"""
        measurements = {}
        
        # Wheel encoder measurements
        if current_time - self.last_wheel_time >= 1.0 / self.config.wheel_encoder_hz:
            # Compute velocities from wheel speeds (what encoders would measure)
            v_linear = (robot.wheel_speed_left + robot.wheel_speed_right) * robot.wheel_radius / 2
            v_angular = (robot.wheel_speed_right - robot.wheel_speed_left) * robot.wheel_radius / robot.wheelbase
            
            # Add noise
            v_linear_meas = v_linear + np.random.normal(0, self.config.wheel_velocity_noise_std)
            v_angular_meas = v_angular + np.random.normal(0, self.config.wheel_angular_noise_std)
            
            measurements['wheels'] = {
                'linear_velocity': v_linear_meas,
                'angular_velocity': v_angular_meas,
                'time': current_time
            }
            self.last_wheel_time = current_time
            
        # IMU Gyroscope measurements
        if current_time - self.last_gyro_time >= 1.0 / self.config.imu_gyro_hz:
            gyro_meas = robot.angular_velocity_body + self.gyro_bias + \
                       np.random.normal(0, self.config.gyro_noise_std, 3)
            
            measurements['gyro'] = {
                'data': gyro_meas,
                'time': current_time
            }
            self.last_gyro_time = current_time
            
            # Update gyro bias (random walk)
            dt = 1.0 / self.config.imu_gyro_hz
            self.gyro_bias += np.random.normal(0, 
                self.config.gyro_bias_walk_std * np.sqrt(dt), 3)
            
        # IMU Accelerometer measurements  
        if current_time - self.last_accel_time >= 1.0 / self.config.imu_accel_hz:
            accel_meas = robot.acceleration_body + self.accel_bias + \
                        np.random.normal(0, self.config.accel_noise_std, 3)
            
            measurements['accel'] = {
                'data': accel_meas,
                'time': current_time
            }
            self.last_accel_time = current_time
            
            # Update accel bias (random walk)
            dt = 1.0 / self.config.imu_accel_hz
            self.accel_bias += np.random.normal(0,
                self.config.accel_bias_walk_std * np.sqrt(dt), 3)
            
        # IMU Orientation measurements (from magnetometer fusion)
        if current_time - self.last_orientation_time >= 1.0 / self.config.imu_orientation_hz:
            orientation_meas = robot.orientation + \
                              np.random.normal(0, self.config.orientation_noise_std)
            
            measurements['orientation'] = {
                'data': orientation_meas,
                'time': current_time
            }
            self.last_orientation_time = current_time
            
        return measurements

class DifferentialDriveEKF:
    """Extended Kalman Filter for differential drive robot state estimation"""
    
    def __init__(self, config: EKFConfig):
        self.config = config
        
        # State vector: [x, y, θ, v_x, v_y, ω_z, b_gx, b_gy, b_gz, b_ax, b_ay]
        self.x = np.zeros(11)
        
        # Add initial state errors if configured
        if config.init_pos_error is not None:
            self.x[0:2] = config.init_pos_error
        self.x[2] = config.init_orientation_error
        
        # Covariance matrix
        self.P = np.diag([
            config.pos_init_std**2, config.pos_init_std**2, config.orientation_init_std**2,
            config.vel_init_std**2, config.vel_init_std**2, config.vel_init_std**2,
            config.gyro_bias_init_std**2, config.gyro_bias_init_std**2, config.gyro_bias_init_std**2,
            config.accel_bias_init_std**2, config.accel_bias_init_std**2
        ])
        
        # Process noise covariance
        self.Q = np.diag([
            config.pos_process_std**2, config.pos_process_std**2, config.orientation_process_std**2,
            config.vel_process_std**2, config.vel_process_std**2, config.vel_process_std**2,
            config.gyro_bias_process_std**2, config.gyro_bias_process_std**2, config.gyro_bias_process_std**2,
            config.accel_bias_process_std**2, config.accel_bias_process_std**2
        ])
        
        # Measurement noise covariances
        self.R_wheels = np.diag([0.05**2, 0.02**2])  # [linear_vel, angular_vel]
        self.R_gyro = np.diag([0.01**2, 0.01**2, 0.01**2])  # [ωx, ωy, ωz]
        self.R_accel = np.diag([0.1**2, 0.1**2])  # [ax, ay] (only x,y used for orientation)
        self.R_orientation = np.array([[0.05**2]])  # [θ]
        
        # Gravity constant
        self.g = 9.81
        
        self.last_predict_time = 0.0
        self.update_count = 0
        
    def predict(self, dt: float):
        """Prediction step using constant velocity model"""
        if dt <= 0:
            return
            
        # State indices for clarity
        x, y, theta = self.x[0], self.x[1], self.x[2]
        vx, vy, omega_z = self.x[3], self.x[4], self.x[5]
        
        # Process model (constant velocity)
        x_new = x + vx * dt
        y_new = y + vy * dt
        theta_new = theta + omega_z * dt
        # Velocities and biases remain constant
        
        # Update state
        self.x[0] = x_new
        self.x[1] = y_new  
        self.x[2] = theta_new
        
        # Wrap angle
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        
        # Jacobian of process model
        F = np.eye(11)
        F[0, 3] = dt  # x depends on vx
        F[1, 4] = dt  # y depends on vy
        F[2, 5] = dt  # θ depends on ωz
        
        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q * dt
        
    def update_wheel_encoders(self, linear_vel: float, angular_vel: float):
        """Update using wheel encoder measurements"""
        # Current state
        theta = self.x[2]
        
        # Measurement model: convert world velocities to body frame
        # h(x) = [vx*cos(θ) + vy*sin(θ), ωz]
        h = np.array([
            self.x[3] * np.cos(theta) + self.x[4] * np.sin(theta),  # forward velocity
            self.x[5]  # angular velocity
        ])
        
        # Measurement
        z = np.array([linear_vel, angular_vel])
        
        # Innovation
        y = z - h
        
        # Jacobian
        H = np.zeros((2, 11))
        H[0, 2] = -self.x[3] * np.sin(theta) + self.x[4] * np.cos(theta)  # ∂h₁/∂θ
        H[0, 3] = np.cos(theta)   # ∂h₁/∂vx
        H[0, 4] = np.sin(theta)   # ∂h₁/∂vy
        H[1, 5] = 1.0             # ∂h₂/∂ωz
        
        # Update
        self._kalman_update(y, H, self.R_wheels)
        
    def update_gyroscope(self, gyro_measurement: np.ndarray):
        """Update using IMU gyroscope measurements"""
        # Measurement model: h(x) = [b_gx, b_gy, ωz + b_gz]
        # For ground robot: ωx ≈ ωy ≈ 0, so gyro measures mostly bias
        h = np.array([
            self.x[6],              # ωx ≈ b_gx
            self.x[7],              # ωy ≈ b_gy  
            self.x[5] + self.x[8]   # ωz + b_gz
        ])
        
        # Innovation
        y = gyro_measurement - h
        
        # Jacobian
        H = np.zeros((3, 11))
        H[0, 6] = 1.0  # gyro_x measures b_gx
        H[1, 7] = 1.0  # gyro_y measures b_gy
        H[2, 5] = 1.0  # gyro_z measures ωz
        H[2, 8] = 1.0  # gyro_z measures b_gz
        
        # Update
        self._kalman_update(y, H, self.R_gyro)
        
    def update_accelerometer(self, accel_measurement: np.ndarray):
        """Update using IMU accelerometer measurements (for orientation)"""
        theta = self.x[2]
        
        # Measurement model: gravity in body frame
        # h(x) = [-g*sin(θ) + b_ax, g*cos(θ) + b_ay]
        h = np.array([
            -self.g * np.sin(theta) + self.x[9],   # accel_x
            self.g * np.cos(theta) + self.x[10]    # accel_y
        ])
        
        # Use only x,y accelerometer for orientation (z is mainly gravity)
        z = accel_measurement[0:2]
        
        # Innovation
        y = z - h
        
        # Jacobian
        H = np.zeros((2, 11))
        H[0, 2] = -self.g * np.cos(theta)  # ∂h₁/∂θ
        H[0, 9] = 1.0                      # ∂h₁/∂b_ax
        H[1, 2] = -self.g * np.sin(theta)  # ∂h₂/∂θ
        H[1, 10] = 1.0                     # ∂h₂/∂b_ay
        
        # Update
        self._kalman_update(y, H, self.R_accel)
        
    def update_orientation(self, orientation_measurement: float):
        """Update using IMU orientation measurement"""
        # Measurement model: h(x) = θ
        h = np.array([self.x[2]])
        z = np.array([orientation_measurement])
        
        # Handle angle wrapping for innovation
        y = z - h
        y[0] = np.arctan2(np.sin(y[0]), np.cos(y[0]))
        
        # Jacobian
        H = np.zeros((1, 11))
        H[0, 2] = 1.0  # measures θ directly
        
        # Update
        self._kalman_update(y, H, self.R_orientation)
        
    def _kalman_update(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray):
        """Generic Kalman filter update step"""
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State update
        self.x += K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(11) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        
        # Wrap orientation
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))
        
        self.update_count += 1

class PerformanceMetrics:
    """Track and calculate performance metrics for the EKF"""
    
    def __init__(self):
        self.position_errors = []
        self.orientation_errors = []
        self.velocity_errors = []
        self.times = []
        self.convergence_time = None
        self.convergence_threshold = 0.1  # meters
        
    def update(self, true_pos: np.ndarray, true_orientation: float, true_vel: np.ndarray,
               est_pos: np.ndarray, est_orientation: float, est_vel: np.ndarray, time: float):
        """Update metrics with current true and estimated states"""
        pos_error = np.linalg.norm(true_pos - est_pos)
        
        # Handle angle wrapping for orientation error
        orientation_error = true_orientation - est_orientation
        orientation_error = np.abs(np.arctan2(np.sin(orientation_error), np.cos(orientation_error)))
        
        vel_error = np.linalg.norm(true_vel - est_vel)
        
        self.position_errors.append(pos_error)
        self.orientation_errors.append(orientation_error)
        self.velocity_errors.append(vel_error)
        self.times.append(time)
        
        # Check convergence
        if self.convergence_time is None and pos_error < self.convergence_threshold:
            self.convergence_time = time
            
    def get_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        if not self.position_errors:
            return {}
            
        pos_errors = np.array(self.position_errors)
        orientation_errors = np.array(self.orientation_errors)
        vel_errors = np.array(self.velocity_errors)
        
        return {
            'pos_rmse': np.sqrt(np.mean(pos_errors**2)),
            'pos_max': np.max(pos_errors),
            'pos_final': pos_errors[-1],
            'orientation_rmse': np.sqrt(np.mean(orientation_errors**2)),
            'orientation_max': np.max(orientation_errors),
            'vel_rmse': np.sqrt(np.mean(vel_errors**2)),
            'convergence_time': self.convergence_time,
            'final_time': self.times[-1]
        }

def run_simulation(pattern='circle', duration=30.0, sensor_config=None, ekf_config=None):
    """Run the complete simulation"""
    
    # Use default configs if not provided
    if sensor_config is None:
        sensor_config = SensorConfig()
    if ekf_config is None:
        ekf_config = EKFConfig()
        ekf_config.init_pos_error = np.array([0.5, -0.3])
    
    # Initialize components
    robot = RobotSimulator(pattern=pattern, duration=duration)
    sensors = SensorSimulator(sensor_config)
    ekf = DifferentialDriveEKF(ekf_config)
    metrics = PerformanceMetrics()
    
    # Data storage
    true_positions = []
    est_positions = []
    true_orientations = []
    est_orientations = []
    true_velocities = []
    est_velocities = []
    times = []
    bias_estimates = []
    
    print(f"Running {pattern} trajectory simulation for {duration} seconds...")
    print(f"Sensor rates - Wheels: {sensor_config.wheel_encoder_hz}Hz, "
          f"IMU Gyro: {sensor_config.imu_gyro_hz}Hz, "
          f"IMU Accel: {sensor_config.imu_accel_hz}Hz")
    
    start_time = time.time()
    
    while robot.step():
        current_time = robot.t
        
        # Prediction step
        dt = current_time - ekf.last_predict_time
        if dt > 0:
            ekf.predict(dt)
            ekf.last_predict_time = current_time
        
        # Get sensor measurements
        measurements = sensors.update(robot, current_time)
        
        # Process measurements
        if 'wheels' in measurements:
            ekf.update_wheel_encoders(
                measurements['wheels']['linear_velocity'],
                measurements['wheels']['angular_velocity']
            )
            
        if 'gyro' in measurements:
            ekf.update_gyroscope(measurements['gyro']['data'])
            
        if 'accel' in measurements:
            ekf.update_accelerometer(measurements['accel']['data'])
            
        if 'orientation' in measurements:
            ekf.update_orientation(measurements['orientation']['data'])
        
        # Store data
        true_positions.append(robot.position.copy())
        est_positions.append(ekf.x[0:2].copy())
        true_orientations.append(robot.orientation)
        est_orientations.append(ekf.x[2])
        true_velocities.append(robot.velocity.copy())
        est_velocities.append(ekf.x[3:5].copy())
        times.append(current_time)
        bias_estimates.append(ekf.x[6:11].copy())
        
        # Update metrics
        metrics.update(robot.position, robot.orientation, robot.velocity,
                      ekf.x[0:2], ekf.x[2], ekf.x[3:5], current_time)
    
    computation_time = time.time() - start_time
    
    # Convert to arrays
    true_positions = np.array(true_positions)
    est_positions = np.array(est_positions)  
    true_orientations = np.array(true_orientations)
    est_orientations = np.array(est_orientations)
    true_velocities = np.array(true_velocities)
    est_velocities = np.array(est_velocities)
    bias_estimates = np.array(bias_estimates)
    
    # Print performance metrics
    perf = metrics.get_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Position RMSE: {perf['pos_rmse']:.3f} m")
    print(f"  Position Max Error: {perf['pos_max']:.3f} m")
    print(f"  Orientation RMSE: {np.degrees(perf['orientation_rmse']):.1f} deg")
    print(f"  Velocity RMSE: {perf['vel_rmse']:.3f} m/s")
    print(f"  Convergence Time: {perf['convergence_time']:.1f} s" if perf['convergence_time'] else "  Did not converge")
    print(f"  EKF Update Rate: {ekf.update_count/duration:.1f} Hz")
    print(f"  Computation Time: {computation_time:.2f} s ({duration/computation_time:.1f}x real-time)")
    
    return {
        'true_positions': true_positions,
        'est_positions': est_positions,
        'true_orientations': true_orientations,
        'est_orientations': est_orientations,
        'true_velocities': true_velocities,
        'est_velocities': est_velocities,
        'times': np.array(times),
        'bias_estimates': bias_estimates,
        'metrics': perf,
        'sensor_config': sensor_config,
        'ekf_config': ekf_config
    }

def plot_results(results: Dict):
    """Create comprehensive visualization of simulation results"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 2D Trajectory Plot
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(results['true_positions'][:, 0], results['true_positions'][:, 1], 
             'b-', linewidth=2, label='True Trajectory')
    ax1.plot(results['est_positions'][:, 0], results['est_positions'][:, 1], 
             'r--', linewidth=2, label='Estimated Trajectory')
    ax1.scatter(results['true_positions'][0, 0], results['true_positions'][0, 1], 
                color='green', s=100, marker='o', label='Start')
    ax1.scatter(results['true_positions'][-1, 0], results['true_positions'][-1, 1], 
                color='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Position Error Over Time
    ax2 = fig.add_subplot(2, 3, 2)
    pos_errors = np.linalg.norm(results['true_positions'] - results['est_positions'], axis=1)
    ax2.plot(results['times'], pos_errors, 'g-', linewidth=2)
    ax2.axhline(y=0.1, color='k', linestyle='--', alpha=0.5, label='Convergence Threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position Error (m)')
    ax2.set_title('Position Error Over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Orientation Comparison
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(results['times'], np.degrees(results['true_orientations']), 
             'b-', label='True Orientation', alpha=0.7)
    ax3.plot(results['times'], np.degrees(results['est_orientations']), 
             'r--', label='Estimated Orientation', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Orientation (deg)')
    ax3.set_title('Orientation Estimation')
    ax3.legend()
    ax3.grid(True)
    
    # Velocity Components
    ax4 = fig.add_subplot(2, 3, 4)
    for i, label in enumerate(['X', 'Y']):
        ax4.plot(results['times'], results['true_velocities'][:, i], 
                 label=f'True V{label}', alpha=0.7)
        ax4.plot(results['times'], results['est_velocities'][:, i], 
                 '--', label=f'Est V{label}', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('Velocity Estimation')
    ax4.legend()
    ax4.grid(True)
    
    # Bias Estimation
    ax5 = fig.add_subplot(2, 3, 5)
    bias_labels = ['Gyro X', 'Gyro Y', 'Gyro Z', 'Accel X', 'Accel Y']
    for i, label in enumerate(bias_labels):
        ax5.plot(results['times'], results['bias_estimates'][:, i], 
                 label=f'{label} Bias')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Bias')
    ax5.set_title('Sensor Bias Estimation')
    ax5.legend()
    ax5.grid(True)
    
    # Performance Metrics Text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    convergence_str = f"{results['metrics']['convergence_time']:.1f} s" if results['metrics']['convergence_time'] is not None else "Did not converge"
    
    metrics_text = f"""Performance Metrics:

Position RMSE: {results['metrics']['pos_rmse']:.3f} m
Position Max Error: {results['metrics']['pos_max']:.3f} m

Orientation RMSE: {np.degrees(results['metrics']['orientation_rmse']):.1f} deg
Orientation Max Error: {np.degrees(results['metrics']['orientation_max']):.1f} deg

Velocity RMSE: {results['metrics']['vel_rmse']:.3f} m/s

Convergence Time: {convergence_str}
Total Simulation Time: {results['metrics']['final_time']:.1f} s

Sensor Configuration:
Wheel Encoder Rate: {results['sensor_config'].wheel_encoder_hz} Hz
IMU Gyro Rate: {results['sensor_config'].imu_gyro_hz} Hz
IMU Accel Rate: {results['sensor_config'].imu_accel_hz} Hz"""
    
    ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run different simulation scenarios"""
    
    scenarios = [
        {
            'name': 'Circle Trajectory',
            'pattern': 'circle',
            'duration': 25.0,
            'sensor_config': SensorConfig(wheel_encoder_hz=50.0),
        },
        {
            'name': 'Figure-8 Pattern',
            'pattern': 'figure8', 
            'duration': 30.0,
            'sensor_config': SensorConfig(wheel_encoder_hz=50.0),
        },
        {
            'name': 'Square Trajectory',
            'pattern': 'square',
            'duration': 25.0,
            'sensor_config': SensorConfig(wheel_encoder_hz=50.0),
        }
    ]
    
    print("Differential Drive Robot EKF Sensor Fusion")
    print("=" * 50)
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")
        print("-" * 40)
        
        results = run_simulation(
            pattern=scenario['pattern'],
            duration=scenario['duration'],
            sensor_config=scenario['sensor_config']
        )
        
        plot_results(results)
        
        if i < len(scenarios) - 1:
            input("\nPress Enter to continue to next scenario...")

if __name__ == "__main__":
    main()
