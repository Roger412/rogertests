import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from unicorn_c_api_wrapper import Unicorn
matplotlib.use("TkAgg")

# Initialize API
unicorn = Unicorn()
print("API Version:", unicorn.get_api_version())

# List devices
devices = unicorn.get_available_devices()
if not devices:
    exit("No devices found.")

print("Available devices:")
for i, serial in enumerate(devices):
    print(f"[{i}] {serial}")

# User selects device
choice = int(input("Select device by index: "))
if choice < 0 or choice >= len(devices):
    exit("Invalid device index.")

# Output CSV name
output_file = input("Enter output CSV filename (e.g., data.csv): ").strip()
if not output_file.endswith(".csv"):
    output_file += ".csv"

this_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(this_dir, "data", output_file)

# Open device and start acquisition
handle = unicorn.open_device(devices[choice])
unicorn.start_acquisition(handle, test_signal=False)
channel_names = unicorn.get_channel_names(handle)

# Initialize data buffers
sampling_rate = 250
eeg_channels = 8
imu_channels = 6
window_size = 250  # 1 second
buffer = np.zeros((window_size, unicorn.UNICORN_TOTAL_CHANNELS_COUNT))
samples = []

# Real-time plot setup: 2 subplots (EEG + IMU)
fig, (ax_eeg, ax_imu) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

eeg_lines = []
for i in range(eeg_channels):
    line, = ax_eeg.plot(np.zeros(window_size), label=channel_names[i])
    eeg_lines.append(line)

imu_lines = []
for i in range(imu_channels):
    line, = ax_imu.plot(np.zeros(window_size), label=channel_names[8 + i])
    imu_lines.append(line)

ax_eeg.set_title("EEG Channels")
ax_imu.set_title("IMU Channels (Accelerometer & Gyroscope)")
ax_imu.set_xlabel("Samples")
ax_eeg.legend(loc='upper right', fontsize='small')
ax_imu.legend(loc='upper right', fontsize='small')
ax_eeg.grid(True)
ax_imu.grid(True)

# State variable
recording = True

# Key event to stop
def on_key(event):
    global recording
    if event.key == ' ':
        recording = False

fig.canvas.mpl_connect("key_press_event", on_key)

# Animation update
def update_plot(frame):
    global buffer, recording
    if not recording:
        plt.close(fig)
        return
    num_scans = 25
    data = unicorn.get_data(handle, num_scans)  # flat list of 170 values
    data = np.reshape(data, (num_scans, unicorn.UNICORN_TOTAL_CHANNELS_COUNT))

    buffer = np.vstack((buffer[num_scans:], data))  # maintain window size
    samples.extend(data)  # accumulate full recording

    eeg_min, eeg_max = np.min(buffer[:, :eeg_channels]), np.max(buffer[:, :eeg_channels])
    imu_min, imu_max = np.min(buffer[:, 8:14]), np.max(buffer[:, 8:14])

    ax_eeg.set_ylim(eeg_min - 5, eeg_max + 5)
    ax_imu.set_ylim(imu_min - 1, imu_max + 1)

    for i, line in enumerate(eeg_lines):
        line.set_ydata(buffer[:, i])
    for i, line in enumerate(imu_lines):
        line.set_ydata(buffer[:, 8 + i])

    return eeg_lines + imu_lines

ani = FuncAnimation(fig, update_plot, interval=1000 / sampling_rate)

plt.tight_layout()
plt.show()

# Stop acquisition and close device
unicorn.stop_acquisition(handle)
unicorn.close_device(handle)

# Save data
samples = np.array(samples)
header = ",".join(channel_names)
np.savetxt(output_file, samples, delimiter=",", fmt="%.6f", header=header, comments="")
print(f"Saved data to {output_file}")
