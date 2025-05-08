import numpy as np
import time
import joblib
from unicorn_c_api_wrapper import Unicorn

# --- CONFIG ---
MODEL_PATH = "models/csp_lda_model.pkl"
WINDOW_LENGTH = 5.0  # seconds MUST BE SAME AS TRAINING DATA TRIAL WINDOW???
SAMPLE_RATE = 250
CHANNEL_COUNT = 8  # adjust if different
LABELS = ["Left", "Right"]  # assumes 0 = left, 1 = right

# Load model
model = joblib.load(MODEL_PATH)

# Init Unicorn
unicorn = Unicorn()
devices = unicorn.get_available_devices()
if not devices:
    exit("No devices found.")
print("Using:", devices[0])
handle = unicorn.open_device(devices[0])
unicorn.start_acquisition(handle, test_signal=False)

print("🔮 Real-time motor imagery decoding started (CTRL+C to quit)...")
samples_needed = int(WINDOW_LENGTH * SAMPLE_RATE)
buffer = []

try:
    while True:
        data = unicorn.get_data(handle, 1)[0]
        buffer.append(data)
        
        if len(buffer) >= samples_needed:
            eeg_window = np.array(buffer[-samples_needed:]).T  # shape: (channels, samples)
            X_input = eeg_window[np.newaxis, :, :]             # add batch dim
            
            pred = model.predict(X_input)[0]
            print("🧠 Detected:", LABELS[pred])
            
            # Slide window (50% overlap)
            buffer = buffer[int(SAMPLE_RATE * WINDOW_LENGTH // 2):]

except KeyboardInterrupt:
    print("👋 Exiting...")
    unicorn.stop_acquisition(handle)
    unicorn.close_device(handle)
