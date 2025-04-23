import socket

# Create a UDP socket to receive state (telemetry) data from the drone
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('192.168.10.2', 8890))  # <- your drone-facing local IP
sock.settimeout(5)     # timeout after 5 seconds

print("📡 Listening for Tello state packets on UDP port 8890...")

try:
    while True:
        data, addr = sock.recvfrom(1024)
        print(f"📦 Received from {addr}: {data.decode()}")
except socket.timeout:
    print("❌ No telemetry packets received — check firewall or drone state.")
finally:
    sock.close()
