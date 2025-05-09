import numpy as np
import time
import joblib
import os
import sys
from scipy.signal import butter, lfilter, iirnotch

from unicorn_c_api_wrapper import Unicorn

# ------------------ CONFIG ------------------ #
MODEL_PATH = "CSP_ATTEMPTS/models/csp_lda_model_attempt3.pkl"
WINDOW_LENGTH = 2.0  # seconds
SAMPLE_RATE = 250
LABELS = ["Left", "Right"]
SAMPLES_PER_GET_DATA = 125
# -------------------------------------------- #

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

# Load model
model = joblib.load(MODEL_PATH)

# Init Unicorn
unicorn = Unicorn()
print("🔌 API Version:", unicorn.get_api_version())
devices = unicorn.get_available_devices()
if not devices:
    exit("❌ No Unicorn devices found.")
for i, dev in enumerate(devices):
    print(f"[{i}] {dev}")
choice = int(input("Select device by index: "))
handle = unicorn.open_device(devices[choice])
channel_names = unicorn.get_channel_names(handle)

# Get EEG indices
eeg_indices = [i for i, name in enumerate(channel_names) if "EEG" in name]
print(f"🧠 Using {len(eeg_indices)} EEG channels:", [channel_names[i] for i in eeg_indices])

unicorn.start_acquisition(handle, test_signal=False)
print("🧠 Real-time motor imagery decoding started (CTRL+C to quit)...")

samples_needed = int(WINDOW_LENGTH * SAMPLE_RATE)
buffer = np.empty((0, len(channel_names)), dtype=np.float32)

try:
    while True:
        data = unicorn.get_data(handle, SAMPLES_PER_GET_DATA)
        data = np.reshape(data, (SAMPLES_PER_GET_DATA, -1))

        # Scale EEG only
        data[:, eeg_indices] *= 0.25

        # Optional: Apply filters only to EEG
        data[:, eeg_indices] = bandpass_filter(data[:, eeg_indices], 1, 30, SAMPLE_RATE)
        data[:, eeg_indices] = notch_filter(data[:, eeg_indices], 60, SAMPLE_RATE)

        buffer = np.vstack((buffer, data))

        if buffer.shape[0] >= samples_needed:
            # 🔥 SEND FULL CHANNELS (15), assuming that's how you trained!
            eeg_window = buffer[-samples_needed:, :8].T  # use only first 8 channels
            X_input = eeg_window[np.newaxis, :, :]      # shape (1, 8, 1250)

            # Use predict_proba instead of predict
            probs = model.predict_proba(X_input)[0]
            pred = np.argmax(probs)

            msg = " | ".join([f"{label}: {p*100:.1f}%" for label, p in zip(LABELS, probs)])
            print(f"🧠 Detected: {LABELS[pred]} — {msg} — Raw: {probs}")

            buffer = buffer[int(SAMPLE_RATE * WINDOW_LENGTH):]

        time.sleep(SAMPLES_PER_GET_DATA / SAMPLE_RATE)

except KeyboardInterrupt:
    print("👋 Exiting...")
    unicorn.stop_acquisition(handle)
    unicorn.close_device(handle)
