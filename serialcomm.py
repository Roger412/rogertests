import serial
import threading
import time
import re

# Configure serial port
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.2)

# Shared variable for desired pose
desired_pose = [1.5, 2.5, 0.5, 0.5, 0.55]
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
        desired_pose = [val + 0.1 for val in desired_pose]

        time.sleep(0.2)


# Receiving thread: listens for STM32 replies
def receiver():
    while True:
        try:
            line = ser.readline().decode().strip()
            if line:
                values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                if len(values) == 8:
                    x_d, y_d, phi_d, d, r, roll, pitch, yaw = values
                    print(f"📥 Received: x={x_d:.2f}, y={y_d:.2f}, phi={phi_d:.2f}, d={d:.2f}, r={r:.2f}, roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
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
