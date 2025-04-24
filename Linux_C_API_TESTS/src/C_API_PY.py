import ctypes
from ctypes import *

# Load Unicorn shared library
lib = ctypes.cdll.LoadLibrary("Linux_C_API_TESTS/lib/libunicorn.so")

# Example: define argument and return types
lib.UNICORN_GetApiVersion.restype = c_float
version = lib.UNICORN_GetApiVersion()
print("Unicorn API Version:", version)
