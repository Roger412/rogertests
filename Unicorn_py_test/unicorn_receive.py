import numpy as np
import os
import matplotlib.pyplot as plt
from unicorn_c_api_wrapper import Unicorn

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

# Open device
handle = unicorn.open_device(devices[choice])
unicorn.start_acquisition(handle, test_signal=True)

# Get channel names
channel_names = unicorn.get_channel_names(handle)

# Acquire samples
samples = []
for _ in range(250):  # 1 second
    data = unicorn.get_data(handle, 1)
    samples.append(data)

unicorn.stop_acquisition(handle)
unicorn.close_device(handle)

# Save to CSV with header
samples = np.array(samples)
header = ','.join(channel_names)
np.savetxt(output_file, samples, delimiter=",", fmt="%.6f", header=header, comments='')
print(f"Saved data to {output_file}")

# Plot EEG and IMU separately
eeg_data = samples[:, :8]
imu_data = samples[:, 8:14]  # ACC X,Y,Z + GYR X,Y,Z

time = np.arange(samples.shape[0]) / 250.0

# EEG Plot
fig1, axs1 = plt.subplots(eeg_data.shape[1], 1, sharex=True, figsize=(10, 8))
for i in range(eeg_data.shape[1]):
    axs1[i].plot(time, eeg_data[:, i])
    axs1[i].set_ylabel(f'EEG {i+1}')
    axs1[i].grid(True)
axs1[-1].set_xlabel('Time (s)')
fig1.suptitle("EEG Channels")

# IMU Plot
labels = ['ACC X', 'ACC Y', 'ACC Z', 'GYR X', 'GYR Y', 'GYR Z']
fig2, axs2 = plt.subplots(imu_data.shape[1], 1, sharex=True, figsize=(10, 6))
for i in range(imu_data.shape[1]):
    axs2[i].plot(time, imu_data[:, i])
    axs2[i].set_ylabel(labels[i])
    axs2[i].grid(True)
axs2[-1].set_xlabel('Time (s)')
fig2.suptitle("IMU Channels (Accelerometer & Gyroscope)")

plt.show()
