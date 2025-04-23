import cv2
import time
from djitellopy import Tello

# Initialize
tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()

frame_read = tello.get_frame_read()

try:
    start = time.time()
    print("Circle + video capture")
    tello.send_rc_control(30, 30, 0, 20)

    while time.time() - start < 6:
        frame = frame_read.frame
        cv2.imshow("Tello Live + Flight", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)

finally:
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    tello.end()
