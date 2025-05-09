import numpy as np
import os

# ------------ CONFIG ------------
SAMPLE_RATE = 250
CLASSES = ["left", "right"]
SAVE_DIR = "CSP_ATTEMPTS/data"
# --------------------------------

filename = input("Enter base filename (without extension): ").strip()
path = os.path.join(SAVE_DIR, filename + ".npz")
data = np.load(path, allow_pickle=True)

samples = data["samples"]  # shape: (n_samples, n_channels)
events = data["events"]    # list of tuples: (label, counter_start, duration_in_seconds)
channel_names = data["channel_names"]
counters = samples[:, -2].astype(int)  # Use Counter column (penultimate)

print("First 10 counters:", counters[:10])
print(f"📂 Loaded {samples.shape[0]} samples and {len(events)} events from {filename}.npz")

X_trials = []
y_labels = []
truncations = []

expected_channels = len(channel_names) - 2  # excluding Battery, Counter, Validation
expected_shapes = []

for i, (label, c_start, duration) in enumerate(events):
    expected_samples = int(duration * SAMPLE_RATE)
    c_end = c_start + expected_samples

    idx_start = np.searchsorted(counters, c_start)
    idx_end = np.searchsorted(counters, c_end)
    segment = samples[idx_start:idx_end, :-2].T  # Remove Battery, Counter, Validation

    actual_samples = segment.shape[1]
    print(f"[ℹ️] Event {i}: '{label}' → expected {expected_samples}, got {actual_samples}")

    if actual_samples < expected_samples:
        print(f"[⚠️] Padding '{label}' with {expected_samples - actual_samples} zeros")
        pad_width = expected_samples - actual_samples
        segment = np.pad(segment, ((0, 0), (0, pad_width)), mode='constant')

    elif actual_samples > expected_samples:
        cut_amount = actual_samples - expected_samples
        print(f"[⚠️] Truncating '{label}' by {cut_amount} samples")
        truncations.append(cut_amount)
        segment = segment[:, :expected_samples]

    # Final check before append
    if segment.shape != (expected_channels, expected_samples):
        print(f"[❌] Skipping event {i}: incorrect final shape {segment.shape}")
        continue

    X_trials.append(segment)
    expected_shapes.append(segment.shape)
    y_labels.append(2 if label in ["idle", "rest"] else CLASSES.index(label))

# ---- Save final data ----
if len(X_trials) == 0:
    print("❌ No valid trials found. Nothing saved.")
else:
    try:
        # Create an empty object array and fill it with 2D arrays
        X = np.empty(len(X_trials), dtype=object)
        for i, trial in enumerate(X_trials):
            X[i] = trial

        y = np.array(y_labels)
        out_path = os.path.join(SAVE_DIR, filename + "_trials.npz")
        np.savez(out_path, X=X, y=y, channel_names=channel_names[:-2])
        print(f"\n✅ Saved {len(X)} variable-length trials to {out_path}")
        print(f"\n📊 Final dataset summary:")
        print(f"   ➤ X shape: {X.shape} (trials, channels, samples)")
        print(f"   ➤ y shape: {y.shape}")
        print(f"   ➤ Class distribution: {[(c, sum(y == i)) for i, c in enumerate(CLASSES + ['idle/rest'])]}")

        # Show truncation summary if any
        if truncations:
            print("\n📉 Truncation summary:")
            print(f"   ➤ Events truncated: {len(truncations)}")
            print(f"   ➤ Max truncation: {max(truncations)} samples")
            print(f"   ➤ Avg truncation: {np.mean(truncations):.2f} samples")
        else:
            print("\n✅ No trials were truncated.")

    except Exception as e:
        print("\n❌ Failed to save trials:")
        raise e
