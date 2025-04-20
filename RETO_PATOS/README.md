# 🦆 Duck Tracking with YOLOv8 and Homography Projection

This project performs object detection and tracking of ducklings using a custom-trained YOLOv8 model, combined with homography projection to map detections into real-world coordinates. The final output is a labeled video with bounding boxes and tracked positions of ducks.

---

## 📁 Project Structure

RETO_PATOS/ ├── patos_dataset/ │ ├── images/ # Original frames (JPEG) │ ├── labels/ # YOLO-format annotations (TXT) │ ├── data.yaml # Dataset config for training │ ├── train/ # Training split │ └── val/ # Validation split ├── homography_matrix/ │ └── homography_matrix.npy # 3x3 matrix from pixel to world coordinates ├── train_yolo.py # Script to launch training ├── prepare_train_yolo.py # Splits dataset and organizes folders ├── patos_sin_sal.mp4 # Original video └── patos_detectados_yolo_entrenado.mp4 # Output video with detections

markdown
Copy
Edit

---

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

