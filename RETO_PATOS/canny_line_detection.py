import cv2
import numpy as np

# Cargar imagen original
img = cv2.imread("RETO_PATOS/frame_00888.jpg")
img_with_lines = img.copy()

# Convertir a escala de grises y detectar bordes
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img, 20, 250)

# Detectar líneas con Hough transform
lines = cv2.HoughLinesP(edges, rho=10, theta=np.pi/180, threshold=20,
                        minLineLength=20, maxLineGap=20)

# Dibujar las líneas detectadas sobre la copia
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Mostrar imagen original con líneas encima
cv2.imshow("Líneas detectadas sobre imagen", img_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
