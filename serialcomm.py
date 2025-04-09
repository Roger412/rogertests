import serial
import time

def main():
    # Abre el puerto serie
    ser = serial.Serial(port='COM11', baudrate=9600, timeout=10)
    print("Serial port opened.")

    # Cadena de prueba que se enviará al STM32
    single_char = "c"
    serial_response = "100.0 50.0 105.0 06.0 11.0\n"

    try:
        while True:
            # Enviar datos
            ser.write(serial_response.encode())
            ser.write(serial_response.encode())
            
            # print(f"Sent: {serial_response.strip()}")

            # Leer una línea desde el STM32
            line = ser.readline().decode(errors='ignore').strip()
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
