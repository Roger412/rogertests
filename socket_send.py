import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 9000))  # bind to any port for reply

sock.sendto(b'command', ('192.168.10.1', 8889))
try:
    response, _ = sock.recvfrom(1024)
    print("✅ Drone replied:", response.decode())
except socket.timeout:
    print("❌ No response from drone.")
sock.close()
