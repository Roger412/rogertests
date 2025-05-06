import numpy as np
import os
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import joblib

### ------------------ CONFIG ------------------ ###

DATA_PATH = os.path.join("data", "training_data.npz")  # Your saved EEG dataset
MODEL_OUTPUT_PATH = os.path.join("models", "csp_lda_model.pkl")
USE_IDLE = False   # Set True to include idle class in training (3-class)
N_COMPONENTS = 4    # Number of CSP components to keep

### -------------------------------------------- ###

# Load data
print("📥 Loading dataset from:", DATA_PATH)
data = np.load(DATA_PATH)
X = data['X']  # shape: (n_trials, n_channels, n_samples)
y = data['y']  # shape: (n_trials,)
channels = data['channel_names']

print("✅ Data loaded: X shape =", X.shape, ", y shape =", y.shape)
print("Classes in dataset:", np.unique(y))

# Optional: remove idle class (label 2)
if not USE_IDLE:
    mask = y < 2
    X = X[mask]
    y = y[mask]
    print("🧹 Removed idle class. Remaining classes:", np.unique(y))

# Define pipeline
print("🛠️ Training CSP + LDA pipeline...")
csp = CSP(n_components=N_COMPONENTS, reg=None, log=True, transform_into='csp_space')
lda = LinearDiscriminantAnalysis()
pipeline = make_pipeline(csp, lda)

# Evaluate model
scores = cross_val_score(pipeline, X, y, cv=5)
print("✅ Cross-validated accuracy: %0.2f ± %0.2f" % (scores.mean(), scores.std()))

# Train on all data
pipeline.fit(X, y)
print("📈 Model trained on all data.")

# Evaluate on training set (just for insight)
y_pred = pipeline.predict(X)
print("\n📊 Classification Report (on training data):")
print(classification_report(y, y_pred))

# Save model
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_OUTPUT_PATH)
print(f"💾 Model saved to {MODEL_OUTPUT_PATH}")
