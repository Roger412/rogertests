import cv2
import numpy as np

# Global for mouse callback
img_dest_copy = None

def get_paste_position(event, x, y, flags, paste_coordinate_list):
    global img_dest_copy
    if event == cv2.EVENT_LBUTTONUP:
        # Draw circle at clicked position
        cv2.circle(img_dest_copy, (x, y), 5, (0, 0, 255), -1)
        paste_coordinate_list.append([x, y])

if __name__ == '__main__':
    # Load source image
    img_src = cv2.imread('single_view_metrology/avocado_eating_avoacdo.webp')
    if img_src is None:
        raise FileNotFoundError("Could not read 'avocado_add.webp'")

    h, w, _ = img_src.shape
    img_src_coordinate = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)

    # Load destination image
    img_dest = cv2.imread('single_view_metrology/billboard.jpg')
    if img_dest is None:
        raise FileNotFoundError("Could not read 'billboard.jpg'")

    # Copy destination for drawing points
    img_dest_copy = img_dest.copy()
    paste_coordinate = []

    cv2.namedWindow('collect coordinate')
    cv2.setMouseCallback('collect coordinate', get_paste_position, paste_coordinate)

    # Main loop for collecting 4 click points
    while True:
        cv2.imshow('collect coordinate', img_dest_copy)
        key = cv2.waitKey(1) & 0xFF
        if len(paste_coordinate) == 4:
            break
        if key == 27:  # ESC to exit
            print("Exited without completing 4 points.")
            exit()

    paste_coordinate = np.array(paste_coordinate, dtype=np.float32)

    # Calculate homography
    matrix, _ = cv2.findHomography(img_src_coordinate, paste_coordinate)
    print(f'Homography matrix:\n{matrix}')

    # Warp source image
    perspective_img = cv2.warpPerspective(img_src, matrix, (img_dest.shape[1], img_dest.shape[0]))

    # Create binary mask from warped image
    gray = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Combine warped image into destination
    result_img = img_dest.copy()
    cv2.copyTo(perspective_img, mask, result_img)

    # Show final images
    cv2.imshow('Warped Image', perspective_img)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
