from djitellopy import Tello
import time

def print_telemetry(tello: Tello):
    try:
        state = tello.get_current_state()
        print(f"📡 Battery: {state.get('bat', '?')}% | Height: {state.get('h', '?')} cm | Temp: {state.get('templ', '?')}°C-{state.get('temph', '?')}°C")
    except Exception as e:
        print(f"⚠️ Error reading telemetry: {e}")

if __name__ == '__main__':
    tello = Tello()

    try:
        tello.connect()
        print("✅ Drone connected")
        print_telemetry(tello)

        tello.takeoff()
        print("🛫 Takeoff complete")

        # 🛰️ Start telemetry loop in background (optional)
        for _ in range(5):
            print_telemetry(tello)
            time.sleep(2)

        # 2.2 Forward-Backward movements
        # tello.move_forward(100)
        # print_telemetry(tello)
        # tello.move_back(100)
        # print_telemetry(tello)

        # 2.3 Height and lateral movements
        # tello.move_up(40)
        # print_telemetry(tello)
        # tello.move_left(80)
        # print_telemetry(tello)
        # tello.move_right(80)
        # print_telemetry(tello)
        # tello.move_down(40)
        # print_telemetry(tello)

        # # 2.4 Rotational movements
        # tello.rotate_clockwise(180)
        # print_telemetry(tello)
        # tello.rotate_counter_clockwise(180)
        # print_telemetry(tello)

        # 2.5 Generate a trajectory (square path)
        print("🌀 Executing square trajectory")
        for _ in range(4):
            tello.move_forward(100)
            tello.rotate_clockwise(90)
            print_telemetry(tello)
            time.sleep(2)

        tello.land()
        print("🛬 Landing complete")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        tello.end()
