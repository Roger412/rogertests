# 🦆 Duck Tracking with YOLOv8 and Homography Projection

This project performs object detection and tracking of ducklings using a custom-trained YOLOv8 model, combined with homography projection to map detections into real-world coordinates. The final output is a labeled video with bounding boxes and tracked positions of ducks.

---

## 📁 Project Structure

```text
RETO_PATOS/
├── patos_dataset/
│   ├── images/                         # Fotogramas originales extraídos del video (.jpg)
│   ├── labels/                         # Anotaciones en formato YOLO (.txt)
│   ├── data.yaml                       # Archivo de configuración para entrenamiento YOLO
│   ├── train/                          # Subconjunto de entrenamiento (imágenes + etiquetas)
│   └── val/                            # Subconjunto de validación (imágenes + etiquetas)
├── homography_matrix/
│   └── homography_matrix.npy          # Matriz 3x3 para transformar píxeles a coordenadas reales
├── prepare_train_yolo.py              # Script para dividir y organizar el dataset en train/val
├── train_yolo.py                      # Script para entrenar YOLOv8 sobre los datos anotados
├── patos_sin_sal.mp4                  # Video original sin anotaciones
└── patos_detectados_yolo_entrenado.mp4 # Video de salida con detecciones de YOLOv8
```
## 🚀 Workflow Overview

### 1. **Image Annotation**

- Used [LabelImg](https://github.com/tzutalin/labelImg) to annotate ducklings in individual video frames.
- Labels saved in YOLO format: `<class> <x_center> <y_center> <width> <height>` (normalized).

### 2. **Dataset Preparation**

- `prepare_train_yolo.py` shuffles and splits annotations/images into `train/` and `val/` folders (80/20 by default).
- Ensures matching `.txt` and `.jpg` files for each sample.
- Generates `data.yaml` describing paths and class names.

### 3. **Training**

- Model: `YOLOv8n` (nano) from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Training config:  
  - `epochs=50`  
  - `imgsz=640`  
  - `batch=16`  
  - Optimizer: `AdamW`
- Model saved to: `runs/detect/train5/weights/best.pt`

### 4. **Homography Calibration**

- Defined homography matrix `H` manually or from keypoint matches.
- Allows projection of detected duck coordinates into real-world space.

### 5. **Inference + Video Output**

- Detects ducks in every video frame using the trained model.
- Projects centroid of each bounding box into world coordinates.
- Annotates each frame with bounding boxes and centroids.
- Saves the output as a new labeled video.

---

## 🛠️ Dependencies

- Python 3.10+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`)
- OpenCV (`pip install opencv-python`)
- NumPy

---

## 🧠 Possible Extensions

- Use DeepSORT for ID-preserving tracking.
- Export duck world-coordinates to CSV for behavioral analysis.
- Generate heatmaps of duck movement.
- Analyze group cohesion or walking patterns.

---

## ✍️ Author

Duck Tracking Research — by Roger 🧠🤖  
Contributions powered by sleepless nights and curious ducklings.

