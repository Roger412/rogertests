import numpy as np
data = np.load("CSP ATTEMPTS/data/t_d_r_3_trials.npz", allow_pickle=True)

X, y = data["X"], data["y"]
print("Trial count:", len(X))
print("Each trial shapes:")
for i, trial in enumerate(X):
    print(f"  Trial {i}: {trial.shape}")