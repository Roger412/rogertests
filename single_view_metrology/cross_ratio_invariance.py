import cv2
import numpy as np

# Globals for callbacks
img_src_copy = None
img_dest_copy = None

def get_src_collinear_points(event, x, y, flags, point_list):
    global img_src_copy
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img_src_copy, (x, y), 5, (0, 255, 0), -1)
        point_list.append([x, y])

def get_paste_points(event, x, y, flags, point_list):
    global img_dest_copy
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img_dest_copy, (x, y), 5, (0, 0, 255), -1)
        point_list.append([x, y])

def cross_ratio_1D(p1, p2, p3, p4):
    return ((p3 - p1) * (p4 - p2)) / ((p3 - p2) * (p4 - p1))

def project_to_line(points):
    direction = points[-1] - points[0]
    direction = direction / np.linalg.norm(direction)
    return np.dot(points - points[0], direction)

def to_homogeneous(points):
    return np.hstack([points, np.ones((points.shape[0], 1))])

def apply_homography(H, points):
    hp = H @ to_homogeneous(points).T
    hp /= hp[2]
    return hp[:2].T

if __name__ == '__main__':
    # === Load images ===
    img_src = cv2.imread('single_view_metrology/avocado_eating_avoacdo.webp')
    if img_src is None:
        raise FileNotFoundError("Could not read 'avocado_eating_avoacdo.webp'")

    img_dest = cv2.imread('single_view_metrology/billboard.jpg')
    if img_dest is None:
        raise FileNotFoundError("Could not read 'billboard.jpg'")

    h, w, _ = img_src.shape
    img_src_coordinate = np.array([[0, 0], [0, h], [w, 0], [w, h]], dtype=np.float32)

    # === Step 1: Get 4 collinear points from source image ===
    img_src_copy = img_src.copy()
    src_collinear_points = []

    print("➡️ Select 4 collinear points on the SOURCE image:")
    cv2.namedWindow('Select 4 collinear source points')
    cv2.setMouseCallback('Select 4 collinear source points', get_src_collinear_points, src_collinear_points)

    while True:
        cv2.imshow('Select 4 collinear source points', img_src_copy)
        key = cv2.waitKey(1) & 0xFF
        if len(src_collinear_points) == 4:
            break
        if key == 27:
            exit("Exited before selecting source points.")

    src_collinear_points = np.array(src_collinear_points, dtype=np.float32)
    projected_before = project_to_line(src_collinear_points)
    cr_before = cross_ratio_1D(*projected_before)
    print(f"\n📏 Cross ratio BEFORE homography: {cr_before:.6f}")

    # === Step 2: Get 4 paste points on destination image ===
    img_dest_copy = img_dest.copy()
    paste_points = []

    print("➡️ Now select 4 corners on DESTINATION image to paste the source image:")
    cv2.namedWindow('Select 4 corners to paste')
    cv2.setMouseCallback('Select 4 corners to paste', get_paste_points, paste_points)

    while True:
        cv2.imshow('Select 4 corners to paste', img_dest_copy)
        key = cv2.waitKey(1) & 0xFF
        if len(paste_points) == 4:
            break
        if key == 27:
            exit("Exited before placing source image.")

    paste_points = np.array(paste_points, dtype=np.float32)

    # === Step 3: Warp source image ===
    H, _ = cv2.findHomography(img_src_coordinate, paste_points)
    print(f"\n📐 Homography matrix:\n{H}")

    perspective_img = cv2.warpPerspective(img_src, H, (img_dest.shape[1], img_dest.shape[0]))

    # Mask & paste
    gray = cv2.cvtColor(perspective_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    result_img = img_dest.copy()
    cv2.copyTo(perspective_img, mask, result_img)

    # === Step 4: Apply homography to original source points ===
    transformed_points = apply_homography(H, src_collinear_points)
    projected_after = project_to_line(transformed_points)
    cr_after = cross_ratio_1D(*projected_after)

    print(f"\n📏 Cross ratio AFTER homography:  {cr_after:.6f}")
    print(f"🔬 Difference:                   {abs(cr_before - cr_after):.6e}")

    # Draw transformed points
    for pt in transformed_points:
        cv2.circle(result_img, tuple(pt.astype(int)), 5, (255, 0, 255), -1)

    # === Display ===
    cv2.imshow('Warped Image', perspective_img)
    cv2.imshow('Result with Transformed Points', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
