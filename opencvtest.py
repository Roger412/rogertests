import cv2

# Load the image (replace 'image.png' with your filename)
image_path = "rogertests/avocado_eating_avoacdo.webp"  # Change this to your image file path
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not read the image.")
else:
    # Convert from BGR to HSV (OpenCV loads images in BGR format)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert from BGR to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Display images
    cv2.imshow("Original Image", image)
    cv2.imshow("HSV Image", hsv_image)
    cv2.imshow("LAB Image", lab_image)

    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the converted images
    cv2.imwrite("hsv_image.png", hsv_image)
    cv2.imwrite("lab_image.png", lab_image)
