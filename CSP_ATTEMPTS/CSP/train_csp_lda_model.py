import numpy as np
import os
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib

### ------------------ CONFIG ------------------ ###

DATA_PATH = "CSP_ATTEMPTS/data/LIVE_EEG_CSP_2_trials.npz"
MODEL_OUTPUT_PATH = os.path.join("CSP_ATTEMPTS/models", "csp_lda_model_attempt3.pkl")
USE_IDLE = False
N_COMPONENTS = 4

### -------------------------------------------- ###

print("📥 Loading dataset from:", DATA_PATH)
data = np.load(DATA_PATH, allow_pickle=True)
X = data['X']  # shape: (n_trials,)
y = data['y']
channels = data['channel_names']

# Filter out idle trials if needed
if not USE_IDLE:
    non_idle_mask = y < 2
    X = X[non_idle_mask]
    y = y[non_idle_mask]
    print("🧹 Removed idle class. Remaining classes:", np.unique(y))

# Crop remaining trials to minimum length
if isinstance(X, np.ndarray) and X.dtype == object:
    min_len = min(trial.shape[1] for trial in X)
    print(f"✂️ Cropping all trials to {min_len} samples and selecting 8 EEG channels")

    # Slice: [:8, :] assumes EEG channels are first 8
    X = np.stack([trial[:8, :min_len] for trial in X])

print("✅ Data ready: X shape =", X.shape, ", y shape =", y.shape)

# Define CSP + LDA pipeline
print("🛠️ Training CSP + LDA pipeline...")
csp = CSP(n_components=N_COMPONENTS, reg=None, log=True, transform_into='average_power')
lda = LinearDiscriminantAnalysis()
pipeline = make_pipeline(csp, lda)

# Evaluate model with cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
print("✅ Cross-validated accuracy: %0.2f ± %0.2f" % (scores.mean(), scores.std()))

# Train on all data
pipeline.fit(X, y)
print("📈 Model trained on all data.")

# Evaluate on training set
y_pred = pipeline.predict(X)
print("\n📊 Classification Report (on training data):")
print(classification_report(y, y_pred))

# Save the trained model
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_OUTPUT_PATH)
print(f"💾 Model saved to {MODEL_OUTPUT_PATH}")
