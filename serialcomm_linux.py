# same code, different port
import serial
import threading
import time
import re

# Configure serial port
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.2)

# Shared variable for desired pose
desired_pose = [6, 3, 0.5, 1.0, 1.0]
count = 0

# Sending thread: sends control goals periodically
def sender():
    global count, desired_pose
    while True:
        # Round values to 2 decimal places before sending
        rounded_pose = [round(val, 2) for val in desired_pose]
        cmd_str = f"{' '.join(map(str, rounded_pose))}\r\n"
        ser.write(cmd_str.encode())
        print(f"📤 [{count}] Sent: {repr(cmd_str)}")
        count += 1

        # Increment original values (not yet rounded)
        # desired_pose = [val + 0.1 for val in desired_pose]

        time.sleep(0.1)


# Receiving thread: listens for STM32 replies
def receiver():
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"🔤 Raw line: {line}")  # Always show what was received
            try:
                # Extract key=value pairs using regex
                matches = re.findall(r'(\w+)=([-+]?\d*\.\d+|\d+)', line)
                data = {key: float(val) for key, val in matches}

                # Debug printout
                print(f"""📥 Parsed Robot State:
    ➤ Desired Pose:         x={data.get('x_desired', 0):.2f}, y={data.get('y_desired', 0):.2f}, phi={data.get('phi_desired', 0):.2f}, d={data.get('d', 0):.2f}, r={data.get('r', 0):.2f}
    ➤ IMU (Euler):          roll={data.get('roll', 0):.2f}, pitch={data.get('pitch', 0):.2f}, yaw={data.get('yaw', 0):.2f}
    ➤ Encoder Counts:       TIM1={int(data.get('TIM1', 0))}, TIM2={int(data.get('TIM2', 0))}, TIM4={int(data.get('TIM4', 0))}, TIM8={int(data.get('TIM8', 0))}
    ➤ Wheel Omegas:         {data.get('Enc_Wheel_Omega1', 0):.3f}, {data.get('Enc_Wheel_Omega2', 0):.3f}, {data.get('Enc_Wheel_Omega3', 0):.3f}, {data.get('Enc_Wheel_Omega4', 0):.3f}
    ➤ Real Speeds (Global): phi_dot={data.get('Inertial_ang_vel_calc', 0):.3f}, x_dot={data.get('Inertial_x_vel_calc', 0):.3f}, y_dot={data.get('Inertial_y_vel_calc', 0):.3f}
    ➤ Odom Position:        phi={data.get('ODOM_phi', 0):.3f}, x={data.get('ODOM_x_pos', 0):.3f}, y={data.get('ODOM_y_pos', 0):.3f}
    ➤ Pose Errors:          dx={data.get('ODOM_Err_x', 0):.3f}, dy={data.get('ODOM_Err_y', 0):.3f}, dphi={data.get('ODOM_Err_phi', 0):.3f}
    ➤ Ctrl Inertial Speeds: x_dot={data.get('Ctrl_Inertial_x_dot', 0):.3f}, y_dot={data.get('Ctrl_Inertial_y_dot', 0):.3f}, phi_dot={data.get('Ctrl_Inertial_phi_dot', 0):.3f}
    ➤ Ctrl Wheel Speeds:    u1={data.get('Ctrl_necc_u1', 0):.3f}, u2={data.get('Ctrl_necc_u2', 0):.3f}, u3={data.get('Ctrl_necc_u3', 0):.3f}, u4={data.get('Ctrl_necc_u4', 0):.3f}
    ➤ Loop Timing:         current={data.get('ts_current', 0):.3f}, previous={data.get('ts_previous', 0):.3f}, delta={data.get('ts_delta', 0):.3f}
    ➤ Ctrl PWM Values:     PWM1={data.get('Ctrl_duty_u1', 0):.3f}, PWM2={data.get('Ctrl_duty_u2', 0):.3f}, PWM3={data.get('Ctrl_duty_u3', 0):.3f}, PWM4={data.get('Ctrl_duty_u4', 0):.3f}
                """)

                # Optional: use the data dict for logging, plotting, etc.

            except Exception as e:
                print("❌ UART RX Error:", e)


# Start both threads
t_send = threading.Thread(target=sender, daemon=True)
t_recv = threading.Thread(target=receiver, daemon=True)

t_send.start()
t_recv.start()

# Keep the main thread alive
while True:
    time.sleep(1)
