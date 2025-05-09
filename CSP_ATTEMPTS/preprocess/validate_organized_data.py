import numpy as np
data = np.load("CSP_ATTEMPTS/data/LIVE_EEG_CSP_2_trials.npz", allow_pickle=True)

X, y = data["X"], data["y"]
print("Trial count:", len(X))
print("Each trial shapes:")
for i, trial in enumerate(X):
    print(f"  Trial {i}: {trial.shape}")