import serial

# Open the port (adjust device and baud as needed)
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=None)

print("Listening on", ser.port)

try:
    while True:
        # readline() will block until it sees a '\n'
        raw = ser.readline()
        if raw:
            # decode with replacement so you never crash on weird bytes
            line = raw.decode('utf-8', errors='replace').rstrip()
            print(f"📥 {line}")
except KeyboardInterrupt:
    print("\nExiting")
    ser.close()
