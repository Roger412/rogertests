from djitellopy import Tello

if __name__ == '__main__':
    tello = Tello()

    try:
        tello.connect()
        print("✅ Drone connected")

        # Get SDK version (djitellopy doesn't expose SDK version directly,
        # but we can query battery or other status to verify connection)
        battery = tello.get_battery()
        print(f"🔋 Battery level: {battery}%")

    except Exception as e:
        print(f"❌ Error connecting to Tello: {e}")
    
    finally:
        tello.end()
