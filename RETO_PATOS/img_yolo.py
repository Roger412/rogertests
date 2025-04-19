from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import torch
print("🧠 Torch version:", torch.__version__)
print("🧠 Torch device:", torch.xpu.get_device_name(0))


# 1. Cargar modelo y ejecutar detección
filename = "frame_00082.jpg"
model = YOLO("yolov8n.pt")
results = model("RETO_PATOS/" + filename, save=True, project="RETO_PATOS/yolo_tests_out/deteccion_patos", name="out_img")

# 2. Obtener ruta exacta del archivo recién generado
save_dir = results[0].save_dir  # ← carpeta correcta (puede ser deteccion_perros, deteccion_perros2, etc.)
output_path = os.path.join(save_dir, filename)

# 3. Leer imagen y convertir BGR → RGB para matplotlib
img = cv2.imread(output_path)
if img is None:
    raise FileNotFoundError(f"❌ No se pudo leer la imagen generada: {output_path}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 4. Mostrar con matplotlib
plt.imshow(img)
plt.title("YOLO Detection")
plt.axis("off")
plt.show()
