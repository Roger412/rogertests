import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
from unicorn_c_api_wrapper import Unicorn

### ------------------ CONFIG ------------------ ###

START_IDLE_TIME_DURATION = 30
TRIAL_DURATION = 15  # seconds per trial
REST_DURATION = 5     # seconds of idle between trials
SAMPLE_RATE = 250     # Hz
N_TRIALS_PER_CLASS = 18
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

# Helper: Show visual arrow cue
def show_arrow(direction='left', duration=2.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.axis('off')
    arrow = '←' if direction == 'left' else '→'
    ax.text(0.5, 0.5, arrow, color='white', fontsize=120, ha='center', va='center')
    plt.ion()
    plt.show()
    plt.pause(duration)
    plt.close()

# Storage for all trials
all_data = []  # each item: [n_channels x n_samples] numpy array
all_labels = []  # 0 for left, 1 for right, 2 for idle

print("\nStarting guided motor imagery data collection with visual cues...\n")

# Randomized trial order
trial_list = [(label, i) for label in range(len(CLASSES)) for i in range(N_TRIALS_PER_CLASS)]
random.shuffle(trial_list)

# Record initial idle
print("🧘 Starting 30-second IDLE baseline recording — remain still and relaxed.")
time.sleep(2)
idle_samples = []
for _ in range(int(START_IDLE_TIME_DURATION * SAMPLE_RATE)):
    data = unicorn.get_data(handle, 1)
    idle_samples.append(data[0])
idle_array = np.array(idle_samples).T  # shape: (channels, samples)
all_data.append(idle_array)
all_labels.append(2)  # Label for idle

for label_idx, trial_num in trial_list:
    class_label = CLASSES[label_idx]
    print(f"\n👉 Trial {trial_num+1}/{N_TRIALS_PER_CLASS} - IMAGINE: {class_label.upper()} HAND")

    # Show visual arrow
    show_arrow(direction=class_label, duration=2.0)

    print("...Recording...")
    trial_samples = []
    for _ in range(int(TRIAL_DURATION * SAMPLE_RATE)):
        data = unicorn.get_data(handle, 1)
        trial_samples.append(data[0])
    trial_array = np.array(trial_samples)  # shape: [samples, channels]
    all_data.append(trial_array.T)         # convert to [channels, samples]
    all_labels.append(label_idx)

    print("...Rest...")
    rest_samples = []
    for _ in range(int(REST_DURATION * SAMPLE_RATE)):
        data = unicorn.get_data(handle, 1)
        rest_samples.append(data[0])
    rest_array = np.array(rest_samples).T
    all_data.append(rest_array)
    all_labels.append(2)  # Label as idle

print("\n✅ Done collecting trials.")
unicorn.stop_acquisition(handle)
unicorn.close_device(handle)

# Convert and save as .npz
X = np.stack(all_data)  # shape: (n_trials, n_channels, n_samples)
y = np.array(all_labels)
np.savez(output_path, X=X, y=y, channel_names=channel_names)
print(f"Saved data to {output_path}")
