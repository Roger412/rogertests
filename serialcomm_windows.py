import serial
import threading
import time
import re

# Configure serial port
ser = serial.Serial('COM11', 115200, timeout=0.2)

# Shared variable for desired pose
desired_pose = [1.5, 2.5, 0.5, 0.5, 0.55]
count = 0

# Sending thread: sends control goals periodically
def sender():
    global count, desired_pose  # ✅ Needed to modify the global list
    while True:
        cmd_str = f"{' '.join(map(str, desired_pose))}\r\n"
        ser.write(cmd_str.encode())
        print(f"📤 [{count}] Sent: {repr(cmd_str)}")
        count += 1

        # ✅ Increment each element of the pose for visual confirmation
        desired_pose = [val + 0.1 for val in desired_pose]

        time.sleep(1.0)

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
