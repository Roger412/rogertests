#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
import serial
from adafruit_bno08x_rvc import BNO08x_RVC
from tf_transformations import euler_from_quaternion
import time

class BNO085ImuPublisher(Node):
    def __init__(self):
        super().__init__('bno085_imu_publisher')
        self.publisher_ = self.create_publisher(Imu, '/bno085/imu', 10)
        self.euler_pub = self.create_publisher(Float32MultiArray, '/bno085/euler', 10)
        self.serial_port = serial.Serial("/dev/serial0", baudrate=115200, timeout=1)
        self.sensor = BNO08x_RVC(self.serial_port)
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100 Hz

    def timer_callback(self):
        quat = self.sensor.quaternion  # Returns (x, y, z, w)
        lin_accel = self.sensor.linear_acceleration  # Returns (x, y, z)
        ang_vel = self.sensor.gyro  # Returns (x, y, z)


        if quat and lin_accel and ang_vel:
            
            # Convert to RPY
            euler_msg = Float32MultiArray()
            roll, pitch, yaw = euler_from_quaternion(quat)
            euler_msg.data = [roll, pitch, yaw]

            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'bno085'

            # Orientation
            imu_msg.orientation.x = quat[0]
            imu_msg.orientation.y = quat[1]
            imu_msg.orientation.z = quat[2]
            imu_msg.orientation.w = quat[3]

            # Angular velocity
            imu_msg.angular_velocity.x = ang_vel[0]
            imu_msg.angular_velocity.y = ang_vel[1]
            imu_msg.angular_velocity.z = ang_vel[2]

            # Linear acceleration
            imu_msg.linear_acceleration.x = lin_accel[0]
            imu_msg.linear_acceleration.y = lin_accel[1]
            imu_msg.linear_acceleration.z = lin_accel[2]

            self.euler_pub.publish(euler_msg)
            self.publisher_.publish(imu_msg)
        else:
            self.get_logger().warn('Failed to read data from BNO085 sensor.')

def main(args=None):
    rclpy.init(args=args)
    node = BNO085ImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
