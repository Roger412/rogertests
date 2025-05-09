import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys
from unicorn_c_api_wrapper import Unicorn
from scipy.signal import butter, lfilter, iirnotch


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

class UnicornLiveViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        mode = input("Select mode: [1] Live Mode, [2] Playback CSV Mode: ").strip()
        if mode == "1":
            self.live_mode = True
        elif mode == "2":
            self.live_mode = False
        else:
            print("Invalid mode selected. Exiting.")
            sys.exit(1)

        self.sampling_rate = 250
        self.eeg_channels = 8
        self.imu_channels = 6
        self.window_size = 250
        self.freqs = np.fft.rfftfreq(self.window_size, d=1.0/self.sampling_rate)
        self.band_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        self.band_limits = {
            "Delta": (0.5, 4),
            "Theta": (4, 8),
            "Alpha": (8, 13),
            "Beta": (13, 30),
            "Gamma": (30, 45)
        }
        self.band_power_history = {band: [] for band in self.band_labels}
        self.max_band_points = 300

        self.recording = True
        self.samples = []

        if self.live_mode:
            self.unicorn = Unicorn()
            print("API Version:", self.unicorn.get_api_version())
            devices = self.unicorn.get_available_devices()
            if not devices:
                exit("No devices found.")

            for i, serial in enumerate(devices):
                print(f"[{i}] {serial}")
            choice = int(input("Select device by index: "))
            if choice < 0 or choice >= len(devices):
                exit("Invalid device index.")

            output_file = input("Enter output CSV filename (e.g., data.csv): ").strip()
            if not output_file.endswith(".csv"):
                output_file += ".csv"
            this_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_file = os.path.join(this_dir, "data", output_file)

            self.handle = self.unicorn.open_device(devices[choice])
            if self.handle is None:
                print("[ERROR] Failed to open Unicorn device.")
                sys.exit(1)
            self.unicorn.start_acquisition(self.handle, test_signal=False)
            self.channel_names = self.unicorn.get_channel_names(self.handle)
            num_channels = len(self.channel_names)
            self.buffer = np.zeros((self.window_size, num_channels))
            self.plot_duration = 5  # seconds
            self.plot_buffer_size = self.plot_duration * self.sampling_rate
            self.plot_buffer = np.zeros((self.plot_buffer_size, self.buffer.shape[1]))


        else:
            filepath = input("Enter path to CSV file: ").strip()
            self.samples = np.loadtxt(filepath, delimiter=",", skiprows=1)
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            self.channel_names = first_line.lstrip('#').split(',')
            print(f"Loaded {self.samples.shape[0]} samples from {filepath}")
            self.output_file = None
            self.sample_index = 0

            num_channels = self.samples.shape[1]
            self.buffer = np.zeros((self.window_size, num_channels))

        self.eeg_indices = [i for i, name in enumerate(self.channel_names) if "EEG" in name]
        self.imu_indices = [i for i, name in enumerate(self.channel_names) if "Accelerometer" in name or "Gyroscope" in name]

        self.init_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000 // 100)

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        title = QtWidgets.QLabel("\U0001f9e0 Unicorn Live Viewer")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        grid = QtWidgets.QGridLayout()
        main_layout.addLayout(grid)

        # EEG LEFT (Channels 1–4)
        eeg_left_section = QtWidgets.QVBoxLayout()
        eeg_left_label = QtWidgets.QLabel("🧠 EEG Channels 1–4")
        eeg_left_label.setAlignment(QtCore.Qt.AlignCenter)
        eeg_left_label.setStyleSheet("font-size: 18px;")
        eeg_left_section.addWidget(eeg_left_label)

        self.eeg_win_left = pg.GraphicsLayoutWidget()
        eeg_left_section.addWidget(self.eeg_win_left)

        # EEG RIGHT (Channels 5–8)
        eeg_right_section = QtWidgets.QVBoxLayout()
        eeg_right_label = QtWidgets.QLabel("🧠 EEG Channels 5–8")
        eeg_right_label.setAlignment(QtCore.Qt.AlignCenter)
        eeg_right_label.setStyleSheet("font-size: 18px;")
        eeg_right_section.addWidget(eeg_right_label)

        self.eeg_win_right = pg.GraphicsLayoutWidget()
        eeg_right_section.addWidget(self.eeg_win_right)

        self.eeg_curves = []

        # First 4 channels (left)
        for i in range(4):
            p = self.eeg_win_left.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True)
            p.addLegend(offset=(10, 10))
            curve = p.plot(pen='y', name=self.channel_names[i])
            self.eeg_curves.append(curve)

        # Next 4 channels (right)
        for i in range(4, 8):
            p = self.eeg_win_right.addPlot(row=i - 4, col=0)
            p.showGrid(x=True, y=True)
            p.addLegend(offset=(10, 10))
            curve = p.plot(pen='y', name=self.channel_names[i])
            self.eeg_curves.append(curve)

        # IMU section
        imu_section = QtWidgets.QVBoxLayout()
        imu_label = QtWidgets.QLabel("\U0001f9ed IMU Channels (Time Domain)")
        imu_label.setAlignment(QtCore.Qt.AlignCenter)
        imu_label.setStyleSheet("font-size: 18px;")
        imu_section.addWidget(imu_label)

        self.imu_win = pg.GraphicsLayoutWidget()
        imu_section.addWidget(self.imu_win)

        self.imu_curves = []
        imu_channel_names = [name for name in self.channel_names if "Accelerometer" in name or "Gyroscope" in name]

        for i in range(self.imu_channels):
            row = i // 2
            col = i % 2
            p = self.imu_win.addPlot(row=row, col=col)
            p.showGrid(x=True, y=True)
            p.addLegend(offset=(10, 10))
            curve = p.plot(pen='g', name=imu_channel_names[i])
            self.imu_curves.append(curve)

        # Bandpower section
        bandpower_section = QtWidgets.QVBoxLayout()
        bandpower_label = QtWidgets.QLabel("\U0001f3a7 EEG Band Powers Over Time")
        bandpower_label.setAlignment(QtCore.Qt.AlignCenter)
        bandpower_label.setStyleSheet("font-size: 18px;")
        bandpower_section.addWidget(bandpower_label)

        self.bandpower_win = pg.GraphicsLayoutWidget()
        bandpower_section.addWidget(self.bandpower_win)
        self.bandpower_plot = self.bandpower_win.addPlot()
        self.bandpower_plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        self.bandpower_plot.showGrid(x=True, y=True)
        self.bandpower_plot.addLegend(offset=(10, 10))

        colors = {'Delta': 'r', 'Theta': 'm', 'Alpha': 'c', 'Beta': 'g', 'Gamma': 'y'}
        self.bandpower_curves = {}
        for band, color in colors.items():
            curve = self.bandpower_plot.plot(pen=color, name=band)
            self.bandpower_curves[band] = curve

        # Add to layout
        grid.addLayout(eeg_left_section, 0, 0)
        grid.addLayout(eeg_right_section, 0, 1)
        grid.addLayout(imu_section, 1, 0)
        grid.addLayout(bandpower_section, 1, 1)

        self.setWindowTitle("\U0001f9e0 Unicorn Live Viewer")
        self.resize(1800, 1000)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.recording = False

    def update_plot(self):
        if not self.recording:
            self.finish()
            return

        if self.live_mode:
            num_scans = 35
            raw = self.unicorn.get_data(self.handle, num_scans)
            num_channels = self.buffer.shape[1]
            raw = np.reshape(raw, (num_scans, num_channels))

            # Extract EEG data (first 8 channels)
            eeg_data = raw[:, self.eeg_indices] * 0.25  # scale raw ADC to µV

            # Apply filters to EEG data
            filtered_eeg = bandpass_filter(eeg_data, 1, 30, self.sampling_rate)
            filtered_eeg = notch_filter(filtered_eeg, 60, self.sampling_rate)

            # Replace EEG part with filtered version
            raw[:, self.eeg_indices] = filtered_eeg

            self.buffer = np.vstack((self.buffer[num_scans:], raw))
            self.samples.extend(raw)
            self.plot_buffer = np.vstack((self.plot_buffer[num_scans:], raw))
            # ---- EEG Plot using plot_buffer ----
            x_vals = np.linspace(-self.plot_duration, 0, self.plot_buffer_size)
            

        else:
            if self.sample_index >= self.samples.shape[0]:
                print("Reached end of file.")
                self.recording = False
                self.finish()
                return

            num_scans = 5
            data = self.samples[self.sample_index:self.sample_index + num_scans, :]
            self.sample_index += num_scans

            if data.shape[0] < num_scans:
                print("Reached end of file.")
                self.recording = False
                self.finish()
                return

            self.buffer = np.vstack((self.buffer[num_scans:], data))

        # ---- EEG Time Domain ----
        for curve, ch_idx in zip(self.eeg_curves, self.eeg_indices):
                curve.setData(x_vals, self.plot_buffer[:, ch_idx])

        # ---- FFT per EEG Channel ----
        fft_mags = [np.abs(np.fft.rfft(self.buffer[:, ch_idx])) for ch_idx in self.eeg_indices]
        # for curve, mag in zip(self.freq_curves, fft_mags):
        #     curve.setData(self.freqs, mag)

        # ---- IMU Time Domain ----
        for curve, ch_idx in zip(self.imu_curves, self.imu_indices):
            curve.setData(self.buffer[:, ch_idx])

        # ---- Band Powers Over Time ----
        for band, (low, high) in self.band_limits.items():
            idx = np.where((self.freqs >= low) & (self.freqs <= high))[0]
            band_powers = []
            for mag in fft_mags:  # one per EEG channel
                band_energy = np.mean(mag[idx] ** 2)
                band_powers.append(band_energy)

            # Now take the log of the average energy across channels
            avg_power = 10 * np.log10(np.maximum(np.mean(band_powers), 1e-15))
            self.band_power_history[band].append(avg_power)
            if len(self.band_power_history[band]) > self.max_band_points:
                self.band_power_history[band] = self.band_power_history[band][-self.max_band_points:]
            history = self.band_power_history[band]
            x_vals = np.arange(len(history))
            self.bandpower_curves[band].setData(x_vals, history)
            if len(x_vals) > 100:
                self.bandpower_plot.setXRange(x_vals[-100], x_vals[-1])


    def finish(self):
        print("Stopping acquisition...")
        self.timer.stop()

        if self.live_mode and hasattr(self, 'handle') and self.handle is not None:
            try:
                self.unicorn.stop_acquisition(self.handle)
                self.unicorn.close_device(self.handle)
                samples_arr = np.array(self.samples)
                np.savetxt(self.output_file, samples_arr, delimiter=",", fmt="%.6f", header=",".join(self.channel_names), comments="")
                print(f"Saved data to {self.output_file}")
            except Exception as e:
                print(f"[Warning] Error while closing device: {e}")

        QtWidgets.QApplication.quit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = UnicornLiveViewer()
    viewer.show()
    sys.exit(app.exec_())
