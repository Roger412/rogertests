import os
import ctypes
from ctypes import *

class UNICORN_AMPLIFIER_CHANNEL(Structure):
    _fields_ = [
        ("name", c_char * 32),
        ("unit", c_char * 32),
        ("range", c_float * 2),
        ("enabled", c_int)
    ]

class UNICORN_AMPLIFIER_CONFIGURATION(Structure):
    _fields_ = [("Channels", UNICORN_AMPLIFIER_CHANNEL * 17)]


class Unicorn:
    def __init__(self):
        # Load .so relative to this file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "lib", "libunicorn.so"))
        self.lib = ctypes.cdll.LoadLibrary(so_path)

        # Types
        self.UNICORN_HANDLE = c_uint64
        self.BOOL = c_int

        # Init constants
        self.UNICORN_SAMPLING_RATE = 250
        self.UNICORN_TOTAL_CHANNELS_COUNT = 17

        # Structs attached to the class

        # Setup function signatures
        self._setup_functions()

        self.UNICORN_AMPLIFIER_CHANNEL = UNICORN_AMPLIFIER_CHANNEL
        self.UNICORN_AMPLIFIER_CONFIGURATION = UNICORN_AMPLIFIER_CONFIGURATION

    def _setup_functions(self):
        lib = self.lib

        lib.UNICORN_GetApiVersion.restype = c_float

        lib.UNICORN_GetAvailableDevices.argtypes = [POINTER(c_char * 14), POINTER(c_uint32), self.BOOL]
        lib.UNICORN_OpenDevice.argtypes = [c_char_p, POINTER(self.UNICORN_HANDLE)]
        lib.UNICORN_StartAcquisition.argtypes = [self.UNICORN_HANDLE, self.BOOL]
        lib.UNICORN_GetData.argtypes = [self.UNICORN_HANDLE, c_uint32, POINTER(c_float), c_uint32]
        lib.UNICORN_StopAcquisition.argtypes = [self.UNICORN_HANDLE]
        lib.UNICORN_CloseDevice.argtypes = [POINTER(self.UNICORN_HANDLE)]

        lib.UNICORN_GetConfiguration.argtypes = [self.UNICORN_HANDLE, c_void_p]
        lib.UNICORN_SetConfiguration.argtypes = [self.UNICORN_HANDLE, c_void_p]
        lib.UNICORN_GetChannelIndex.argtypes = [self.UNICORN_HANDLE, c_char_p, POINTER(c_uint32)]
        lib.UNICORN_GetNumberOfAcquiredChannels.argtypes = [self.UNICORN_HANDLE, POINTER(c_uint32)]
        lib.UNICORN_GetDeviceInformation.argtypes = [self.UNICORN_HANDLE, c_void_p]
        lib.UNICORN_GetDigitalOutputs.argtypes = [self.UNICORN_HANDLE, POINTER(c_uint8)]
        lib.UNICORN_SetDigitalOutputs.argtypes = [self.UNICORN_HANDLE, c_uint8]
        lib.UNICORN_GetLastErrorText.restype = c_char_p

    # Example: API wrapper functions

    def get_api_version(self):
        return self.lib.UNICORN_GetApiVersion()

    def get_last_error(self):
        return self.lib.UNICORN_GetLastErrorText().decode()

    def get_available_devices(self):
        count = c_uint32()
        self.lib.UNICORN_GetAvailableDevices(None, byref(count), True)
        if count.value == 0:
            return []

        serials = (c_char * 14 * count.value)()
        self.lib.UNICORN_GetAvailableDevices(serials, byref(count), True)
        return [serials[i].value.decode() for i in range(count.value)]

    def open_device(self, serial):
        handle = self.UNICORN_HANDLE()
        result = self.lib.UNICORN_OpenDevice(serial.encode(), byref(handle))
        if result != 0:
            raise RuntimeError(f"Open failed: {self.get_last_error()}")
        return handle

    def close_device(self, handle):
        self.lib.UNICORN_CloseDevice(byref(handle))

    def start_acquisition(self, handle, test_signal=True):
        self.lib.UNICORN_StartAcquisition(handle, self.BOOL(test_signal))

    def stop_acquisition(self, handle):
        self.lib.UNICORN_StopAcquisition(handle)

    def get_data(self, handle, num_scans):
        num_channels = self.UNICORN_TOTAL_CHANNELS_COUNT
        buffer_len = num_channels * num_scans
        buffer = (c_float * buffer_len)()
        result = self.lib.UNICORN_GetData(handle, num_scans, buffer, buffer_len * sizeof(c_float))
        if result != 0:
            raise RuntimeError(f"GetData failed: {self.get_last_error()}")
        return [buffer[i] for i in range(buffer_len)]

    def get_channel_names(self, handle):
        config = self.UNICORN_AMPLIFIER_CONFIGURATION()
        result = self.lib.UNICORN_GetConfiguration(handle, byref(config))
        if result != 0:
            raise RuntimeError(f"GetConfiguration failed: {self.get_last_error()}")

        return [config.Channels[i].name.decode() for i in range(self.UNICORN_TOTAL_CHANNELS_COUNT)]
