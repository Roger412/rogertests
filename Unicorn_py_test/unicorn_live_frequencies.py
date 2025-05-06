import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys
from unicorn_c_api_wrapper import Unicorn

class UnicornLiveViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Choose live mode or playback
        mode = input("Select mode: [1] Live Mode, [2] Playback CSV Mode: ").strip()
        if mode == "1":
            self.live_mode = True
        elif mode == "2":
            self.live_mode = False
        else:
            print("Invalid mode selected. Exiting.")
            sys.exit(1)

        # Buffers
        self.sampling_rate = 250
        self.eeg_channels = 8
        self.imu_channels = 6
        self.window_size = 250  # 1 second
        self.freqs = np.fft.rfftfreq(self.window_size, d=1.0/self.sampling_rate)
        self.recording = True
        self.samples = []

        if self.live_mode:
            # Live device
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

            # Output file
            output_file = input("Enter output CSV filename (e.g., data.csv): ").strip()
            if not output_file.endswith(".csv"):
                output_file += ".csv"
            this_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_file = os.path.join(this_dir, "data", output_file)

            # Connect to device
            self.handle = self.unicorn.open_device(devices[choice])
            if self.handle is None:
                print("[ERROR] Failed to open Unicorn device.")
                sys.exit(1)
            self.unicorn.start_acquisition(self.handle, test_signal=False)
            self.channel_names = self.unicorn.get_channel_names(self.handle)
            
            # 🔥 Set correct buffer shape based on real number of channels
            num_channels = len(self.channel_names)
            self.buffer = np.zeros((self.window_size, num_channels))

        else:
            # Playback mode
            filepath = input("Enter path to CSV file: ").strip()
            self.samples = np.loadtxt(filepath, delimiter=",", skiprows=1)
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
            self.channel_names = first_line.lstrip('#').split(',')
            print(f"Loaded {self.samples.shape[0]} samples from {filepath}")
            self.output_file = None  # no saving in playback mode
            self.sample_index = 0

            # 🔥 Correct buffer shape
            num_channels = self.samples.shape[1]
            self.buffer = np.zeros((self.window_size, num_channels))
            self.eeg_channels = sum(1 for name in self.channel_names if "EEG" in name)
            self.imu_channels = sum(1 for name in self.channel_names if "Accelerometer" in name or "Gyroscope" in name)


        # Build GUI
        self.init_ui()

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(1000 // self.sampling_rate)

    def init_ui(self):
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Main vertical layout
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title at top
        title = QtWidgets.QLabel("🧠 Unicorn Live Viewer")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        main_layout.addWidget(title)

        # Grid layout for the 2x2 sections
        grid = QtWidgets.QGridLayout()
        main_layout.addLayout(grid)

        # ---------------- TOP LEFT (EEG Time Domain) ----------------
        eeg_section = QtWidgets.QVBoxLayout()
        eeg_label = QtWidgets.QLabel("📈 EEG Channels (Time Domain)")
        eeg_label.setAlignment(QtCore.Qt.AlignCenter)
        eeg_label.setStyleSheet("font-size: 18px;")
        eeg_section.addWidget(eeg_label)

        self.eeg_win = pg.GraphicsLayoutWidget()
        eeg_section.addWidget(self.eeg_win)

        self.eeg_curves = []
        for i in range(self.eeg_channels):
            p = self.eeg_win.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True)
            curve = p.plot(pen='y')
            self.eeg_curves.append(curve)

        # ---------------- BOTTOM LEFT (IMU Time Domain) ----------------
        imu_section = QtWidgets.QVBoxLayout()
        imu_label = QtWidgets.QLabel("🧭 IMU Channels (Time Domain)")
        imu_label.setAlignment(QtCore.Qt.AlignCenter)
        imu_label.setStyleSheet("font-size: 18px;")
        imu_section.addWidget(imu_label)

        self.imu_win = pg.GraphicsLayoutWidget()
        imu_section.addWidget(self.imu_win)

        self.imu_curves = []
        for i in range(self.imu_channels):
            p = self.imu_win.addPlot(row=i, col=0)
            p.showGrid(x=True, y=True)
            curve = p.plot(pen='g')
            self.imu_curves.append(curve)

        # ---------------- TOP RIGHT (EEG Frequency Domain) ----------------
        freq_section = QtWidgets.QVBoxLayout()
        freq_label = QtWidgets.QLabel("📊 FFT per EEG Channel (Frequency Domain)")
        freq_label.setAlignment(QtCore.Qt.AlignCenter)
        freq_label.setStyleSheet("font-size: 18px;")
        freq_section.addWidget(freq_label)

        self.freq_win = pg.GraphicsLayoutWidget()
        freq_section.addWidget(self.freq_win)

        self.freq_curves = []
        self.freqs = np.fft.rfftfreq(self.window_size, d=1.0/self.sampling_rate)
        for i in range(self.eeg_channels):
            p = self.freq_win.addPlot(row=i, col=0)
            p.setXRange(0, 60)
            p.showGrid(x=True, y=True)
            curve = p.plot(pen='c')
            self.freq_curves.append(curve)

        # ---------------- BOTTOM RIGHT (EMPTY) ----------------
        empty_widget = QtWidgets.QLabel("")
        empty_widget.setMinimumSize(400, 400)

        # ---------------- Add everything to Grid ----------------
        grid.addLayout(eeg_section, 0, 0)  # row=0, col=0
        grid.addLayout(freq_section, 0, 1) # row=0, col=1
        grid.addLayout(imu_section, 1, 0)  # row=1, col=0
        grid.addWidget(empty_widget, 1, 1) # row=1, col=1

        self.setWindowTitle("🧠 Unicorn Live Viewer")
        self.resize(1800, 1000)


    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.recording = False

    def update_plot(self):
        if not self.recording:
            self.finish()
            return

        if self.live_mode:
            num_scans = 25
            data = self.unicorn.get_data(self.handle, num_scans)
            num_channels = self.buffer.shape[1]
            data = np.reshape(data, (num_scans, num_channels))
            self.buffer = np.vstack((self.buffer[num_scans:], data))
            self.samples.extend(data)
        else:
            # Read 1 scan at a time from CSV
            # Playback mode
            if self.sample_index >= self.samples.shape[0]:
                print("Reached end of file.")
                self.recording = False
                self.finish()
                return
            
            num_scans = 5
            data = self.samples[self.sample_index:self.sample_index + num_scans, :]
            self.sample_index += num_scans

            # If last chunk is smaller, pad it or stop
            if data.shape[0] < num_scans:
                print("Reached end of file.")
                self.recording = False
                self.finish()
                return

            self.buffer = np.vstack((self.buffer[num_scans:], data))


        # Update EEG time domain
        for i, curve in enumerate(self.eeg_curves):
            curve.setData(self.buffer[:, i])

        # Update IMU time domain
        for i, curve in enumerate(self.imu_curves):
            idx = 8 + i
            curve.setData(self.buffer[:, idx])

        # Update EEG frequency domain
        for i, curve in enumerate(self.freq_curves):
            fft_mag = np.abs(np.fft.rfft(self.buffer[:, i]))
            curve.setData(self.freqs, fft_mag)


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
