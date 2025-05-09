import numpy as np
import os
import time
import threading
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
import sys
from scipy.signal import butter, lfilter, iirnotch

# Add parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unicorn_c_api_wrapper import Unicorn

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def notch_filter(data, notch_freq, fs, quality_factor=30):
    nyq = 0.5 * fs
    freq = notch_freq / nyq
    b, a = iirnotch(freq, quality_factor)
    return lfilter(b, a, data, axis=0)

# ------------ CONFIG ------------
SAMPLE_RATE = 250
CLASSES = ["left", "right"]
TRIAL_DURATION = 10
REST_DURATION = 3
IDLE_DURATION = 10
TRIALS_PER_CLASS = 10
SAMPLES_PER_GET_DATA = 55
SAVE_DIR = "CSP_ATTEMPTS/data"
ASSETS_DIR = "assets"
os.makedirs(SAVE_DIR, exist_ok=True)
# --------------------------------

# Shared buffers
sample_buffer = []
timestamp_buffer = []
events = []
running = True
buffer_lock = threading.Lock()

# Initialize Unicorn
print("Initializing Unicorn...")
unicorn = Unicorn()
print("API Version:", unicorn.get_api_version())
devices = unicorn.get_available_devices()
if not devices:
    exit("No Unicorn devices found.")
for i, serial in enumerate(devices):
    print(f"[{i}] {serial}")
choice = int(input("Select device by index: "))
handle = unicorn.open_device(devices[choice])
channel_names = unicorn.get_channel_names(handle)

# Background acquisition thread
def acquisition_loop():
    global running
    unicorn.start_acquisition(handle, test_signal=False)
    print("🧠 Acquisition started.")

    eeg_indices = [i for i, name in enumerate(channel_names) if "EEG" in name]

    while running:
        data = unicorn.get_data(handle, SAMPLES_PER_GET_DATA)  # shape: (SAMPLES_PER_GET_DATA, num_channels)
        data = np.reshape(data, (SAMPLES_PER_GET_DATA, -1))

        # Scale raw EEG (convert to µV)
        data[:, eeg_indices] = data[:, eeg_indices] * 0.25

        # Apply 1–30 Hz bandpass
        data[:, eeg_indices] = bandpass_filter(data[:, eeg_indices], 1, 30, SAMPLE_RATE)

        # Apply 60 Hz notch
        data[:, eeg_indices] = notch_filter(data[:, eeg_indices], 60, SAMPLE_RATE)

        with buffer_lock:
            for row in data:
                sample_buffer.append(row)  # includes EEG, IMU, Battery, Counter, Validation

        time.sleep(SAMPLES_PER_GET_DATA / SAMPLE_RATE)

    unicorn.stop_acquisition(handle)
    unicorn.close_device(handle)
    print("🧠 Acquisition stopped.")


def show_arrow(direction='left', duration=2.0):
    img_path = os.path.join(os.path.dirname(__file__), f"{ASSETS_DIR}/{direction}.png")
    if not os.path.exists(img_path):
        print(f"[Warning] Missing image: {img_path}")
        return
    img = mpimg.imread(img_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis('off')
    plt.ion()
    plt.show()
    plt.pause(duration)
    plt.close()

def log_event(label, duration):
    with buffer_lock:
        last_counter = int(sample_buffer[-1][15]) if sample_buffer else -1  # 15 = Counter index
    events.append((label, last_counter, duration))

# ------------------------ RUN ------------------------
filename = input("Enter base filename to save (e.g., session1): ").strip()
if not filename:
    filename = "session"

acq_thread = threading.Thread(target=acquisition_loop)
acq_thread.start()

# IDLE
time.sleep(2)
print(f"🧘 IDLE for {IDLE_DURATION}s")
log_event("idle", IDLE_DURATION)
time.sleep(IDLE_DURATION)

# Trials
trial_list = [(cls, i) for cls in CLASSES for i in range(TRIALS_PER_CLASS)]
np.random.shuffle(trial_list)

for cls, i in trial_list:
    print(f"👉 {cls.upper()} HAND")
    log_event(cls, TRIAL_DURATION)
    show_arrow(cls, TRIAL_DURATION)
    print("...Rest...")
    log_event("rest", REST_DURATION)
    time.sleep(REST_DURATION)

# Stop
print("✅ Trials complete.")
running = False
acq_thread.join()

# Save raw data
samples = np.array(sample_buffer)

np.savez(os.path.join(SAVE_DIR, filename + ".npz"),
         samples=samples,
         events=np.array(events, dtype=object),
         channel_names=channel_names)
print(f"💾 Saved raw data to {filename}.npz")
