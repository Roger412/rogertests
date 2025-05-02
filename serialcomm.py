import serial
import threading
import time
import re

# Configure serial port
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.2)

# Shared variable for desired pose
desired_pose = [1.0, 2.0, 0.0, 0.3, 0.05]

# Sending thread: sends control goals periodically
def sender():
    while True:
        cmd_str = f"{desired_pose[0]} {desired_pose[1]} {desired_pose[2]} {desired_pose[3]} {desired_pose[4]} "
        ser.write(cmd_str.encode())
        print(f"📤 Sent: {cmd_str.strip()}")
        time.sleep(0.2)  # Wait before next command

# Receiving thread: listens for STM32 replies
def receiver():
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                print(f"📥 Received: {line}")
                # Extract numbers using regex
                values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                if len(values) == 5:
                    x_d, y_d, phi_d, d, r = values
                    print(f"🧭 Parsed: x={x_d}, y={y_d}, phi={phi_d}, d={d}, r={r}")
                else:
                    print("⚠️ Unexpected format:", line)
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
