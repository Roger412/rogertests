# 🧠 Motor Imagery BCI with Unicorn

This project attempts to implement a basic EEG-based BCI (Brain-Computer Interface) using the Unicorn Hybrid Black device. It uses CSP + LDA to classify left vs right motor imagery in real time.

---

## ✅ TODO
- [ ] PONERMELO Y PROBAR
- [ ] GUI para ver en tiempo real resultados de modelo

Sugerencias de chat:
- [ ] Add 8–30 Hz bandpass filtering
- [ ] Collect more data (≥15 trials/class)
- [ ] Add live performance visualization
- [ ] Integrate ROS2 robot control
- [ ] Evaluate FBCSP or Riemannian classifiers

---

## 📥 Data Acquisition

EEG data is collected via the Unicorn C API and saved to `.npz` format. Each sample contains:

- 8 EEG channels
- 6 IMU channels
- 3 additional: Battery, Counter, Validation

Sampling rate: **250 Hz**

Each trial is annotated with:

- `label`: 0 (left), 1 (right), 2 (idle)
- `counter`: the EEG sample counter at trial start
- `duration`: trial length in seconds

### File structure of `.npz`:
- `samples`: shape `(N, 17)`
- `events`: list of tuples `(label, counter, duration)`
- `channel_names`: 17 strings

---

## 🧹 Data Organization

A preprocessing step converts `.npz` recordings into:
- Fixed-length EEG trials: shape `(n_trials, n_channels, n_samples)`
- Labels: shape `(n_trials,)`

Idle class (`label == 2`) can optionally be removed (`USE_IDLE=False`).
All remaining trials are cropped to the **minimum trial length** (e.g., 500 samples) for uniform shape.

The resulting files are saved as: `*_trials.npz`

---

## 🧠 Model Training

The training script builds and evaluates a **CSP + LDA pipeline** using scikit-learn and MNE.

### Training steps:
1. Load preprocessed trials: `X`, `y`
2. Remove idle trials (if configured)
3. Crop all trials to equal length
4. Build `Pipeline(CSP → LDA)`
5. Evaluate with 5-fold cross-validation
6. Train on full dataset
7. Save model to `.pkl`

```bash
python train_csp_rename.py
```

Model is saved as:
```
CSP ATTEMPTS/models/csp_lda_model.pkl
```

---

## 🔮 Real-Time Prediction

Use the trained model to classify live EEG input:

```bash
python live_predictor.py
```

This script:
- Streams EEG from Unicorn in real time
- Collects a 3-second window (750 samples)
- Runs prediction every 1.5 seconds (50% overlap)
- Prints: `🧠 Detected: Left` or `Right`

Ensure the number of channels and window size matches what was used in training (e.g., 8 channels × 500 samples).

---

## 📂 Project Structure (simplified)

```
CSP ATTEMPTS/
├── data/
│   └── t_d_r_3_trials.npz         # raw or preprocessed data
├── models/
│   └── csp_lda_model.pkl          # saved trained model
├── train_csp_rename.py            # training script
└── live_predictor.py              # real-time classification
```

---

## 📌 Requirements

- Python 3.8+
- `numpy`, `scikit-learn`, `mne`, `joblib`
- Unicorn C API Python wrapper

---

## 🧪 Example Usage

Train a new model:
```bash
python train_csp_rename.py
```

Run online classification:
```bash
python live_predictor.py
```

---

## ✍️ Notes

- CSP assumes trials have fixed duration — inconsistent input length will break the model.
- CSP uses class-wise variance patterns to extract features → works well with motor imagery EEG.
- LDA is fast and interpretable, suitable for low-trial EEG classification tasks.

---
