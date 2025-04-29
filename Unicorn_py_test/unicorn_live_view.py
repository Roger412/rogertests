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

# Real-time plot setup: one subplot per channel
fig, axes = plt.subplots(14, 1, figsize=(10, 16), sharex=True)

ax_eeg_list = axes[:8]  # EEG axes
ax_imu_list = axes[8:]  # IMU axes

# EEG lines
eeg_lines = []
for i, ax in enumerate(ax_eeg_list):
    line, = ax.plot(np.zeros(window_size))
    ax.set_ylabel(channel_names[i], fontsize=8)
    ax.grid(True)
    eeg_lines.append(line)
ax_eeg_list[-1].set_xlabel("Samples")

# IMU lines
imu_lines = []
for i, ax in enumerate(ax_imu_list):
    line, = ax.plot(np.zeros(window_size))
    ax.set_ylabel(channel_names[8 + i], fontsize=8)
    ax.grid(True)
    imu_lines.append(line)
ax_imu_list[-1].set_xlabel("Samples")

# State variable
recording = True


fig.suptitle("EEG + IMU Channels")
fig.tight_layout()

# Key event
def on_key(event):
    global recording
    if event.key == ' ':
        recording = False

fig.canvas.mpl_connect("key_press_event", on_key)

def update_plot(frame):
    global buffer, recording
    if not recording:
        plt.close(fig)
        return

    num_scans = 25
    data = unicorn.get_data(handle, num_scans)
    data = np.reshape(data, (num_scans, unicorn.UNICORN_TOTAL_CHANNELS_COUNT))

    buffer = np.vstack((buffer[num_scans:], data))
    samples.extend(data)

    # Update EEG plots
    for i, line in enumerate(eeg_lines):
        line.set_ydata(buffer[:, i])
        ymin, ymax = np.min(buffer[:, i]), np.max(buffer[:, i])
        if ymax - ymin > 1e-6:  # Only if signal has variation
            ax_eeg_list[i].set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

    # Update IMU plots
    for i, line in enumerate(imu_lines):
        idx = 8 + i
        line.set_ydata(buffer[:, idx])
        ymin, ymax = np.min(buffer[:, idx]), np.max(buffer[:, idx])
        if ymax - ymin > 1e-6:
            ax_imu_list[i].set_ylim(ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax))

    return eeg_lines + imu_lines


ani = FuncAnimation(fig, update_plot, interval=1000 / sampling_rate, cache_frame_data=False)

plt.get_current_fig_manager().toolbar.pan()
plt.show()


# Stop acquisition and close device
unicorn.stop_acquisition(handle)
unicorn.close_device(handle)

# Save data
samples = np.array(samples)
header = ",".join(channel_names)
np.savetxt(output_file, samples, delimiter=",", fmt="%.6f", header=header, comments="")
print(f"Saved data to {output_file}")
