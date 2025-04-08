import serial
import time

def main():
    # Abre el puerto serie
    ser = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=10)
    print("Serial port opened.")

    # Cadena de prueba que se enviará al STM32
    serial_response = "10.0 5.0 15.0 6.0 1.0\n"

    try:
        while True:
            # Enviar datos
            # ser.write(serial_response.encode())
            # ser.write(serial_response.encode())
            print(f"Sent: {serial_response.strip()}")

            # Leer una línea desde el STM32
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print(f"Received: {line}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
