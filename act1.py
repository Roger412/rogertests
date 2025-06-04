from djitellopy import Tello
import time

if __name__ == '__main__':
    try:
        # Connect to drone
        # tello = Tello(host="192.168.12.6")
        tello = Tello()
        tello.connect()
        print("✅ Connected to Tello")

        # Optional: show initial battery level
        battery = tello.get_battery()
        print(f"🔋 Battery level: {battery}%")

        tello.send_command_with_return("command")
        time.sleep(0.5)

        # Takeoff
        tello.takeoff()
        print("🛫 Tello took off")
        print(tello.get_current_state())

        # Hover and print telemetry for 10 seconds
        for i in range(10):
            state = tello.get_current_state()
            print(f"""
🛰️ Telemetry Tick {i+1}:
- Battery: {state.get('bat', '?')}%
- Temp: {state.get('templ', '?')}°C to {state.get('temph', '?')}°C
- Height: {state.get('h', '?')} cm
- Pitch/Roll/Yaw: {state.get('pitch', '?')} / {state.get('roll', '?')} / {state.get('yaw', '?')}
- Speed (vgx, vgy, vgz): {state.get('vgx', '?')} / {state.get('vgy', '?')} / {state.get('vgz', '?')}
""")
            time.sleep(1)

        # Land
        tello.land()
        print("🛬 Tello landed")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        tello.end()
