import numpy as np
import os
import time
from unicorn_c_api_wrapper import Unicorn

### ------------------ CONFIG ------------------ ###

TRIAL_DURATION = 2  # seconds per trial
SAMPLE_RATE = 250   # Hz
N_TRIALS_PER_CLASS = 20
CLASSES = ["left", "right"]  # Motor imagery classes
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

### -------------------------------------------- ###

os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize API
unicorn = Unicorn()
print("API Version:", unicorn.get_api_version())

# List devices
devices = unicorn.get_available_devices()
if not devices:
    exit("No Unicorn devices found.")

print("Available devices:")
for i, serial in enumerate(devices):
    print(f"[{i}] {serial}")

# User selects device
choice = int(input("Select device by index: "))
if choice < 0 or choice >= len(devices):
    exit("Invalid device index.")

# Ask for output file
output_file = input("Enter output filename (e.g., training_data.npz): ").strip()
if not output_file.endswith(".npz"):
    output_file += ".npz"
output_path = os.path.join(SAVE_DIR, output_file)

# Open device
handle = unicorn.open_device(devices[choice])
unicorn.start_acquisition(handle, test_signal=False)
channel_names = unicorn.get_channel_names(handle)

# Storage for all trials
all_data = []  # each item: [n_channels x n_samples] numpy array
all_labels = []  # 0 for left, 1 for right

print("\nStarting guided motor imagery data collection...\n")

for label_idx, class_label in enumerate(CLASSES):
    for trial_num in range(N_TRIALS_PER_CLASS):
        print(f"\n👉 Trial {trial_num+1}/{N_TRIALS_PER_CLASS} - IMAGINE: {class_label.upper()} HAND")
        time.sleep(2)  # get ready prompt
        print("...Recording...")
        trial_samples = []

        for _ in range(int(TRIAL_DURATION * SAMPLE_RATE)):
            data = unicorn.get_data(handle, 1)
            trial_samples.append(data[0])

        trial_array = np.array(trial_samples)  # shape: [samples, channels]
        all_data.append(trial_array.T)         # convert to [channels, samples]
        all_labels.append(label_idx)

print("\n✅ Done collecting trials.")
unicorn.stop_acquisition(handle)
unicorn.close_device(handle)

# Convert and save as .npz
X = np.stack(all_data)  # shape: (n_trials, n_channels, n_samples)
y = np.array(all_labels)
np.savez(output_path, X=X, y=y, channel_names=channel_names)
print(f"Saved data to {output_path}")