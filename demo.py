"""
Using cuda.parallel to operate on structs ("dataclasses") on the GPU.
"""

import numpy as np
import cupy as cp

import cuda.parallel.experimental as cudax
from gpudataclass import gpudataclass


# The @gpudataclass decorator registers `Pixel` as a user-defined
# numba type.
@gpudataclass
class Pixel:
    r: np.dtype("int32")
    g: np.dtype("int32")
    b: np.dtype("int32")

# This is the comparator we want to pass to `reduce`. It takes
# two Pixel objects as input and returns the one with the
# larger `g` component as output:
def max_g_value(x, y):
    return x if x.g > y.g else y
    
# Next, we need to initialize data on the device. We'll construct
# a CuPy array of size (10, 3) to represent 10 RGB values
# and view it as a structured dtype:
dtype = np.dtype([("r", "int32"), ("g", "int32"), ("b", "int32")])
d_rgb = cp.random.randint(0, 256, (10, 3), dtype=cp.int32).view(dtype)

# Create an empty array to store the output:
d_out = cp.zeros(1, dtype)

# The initial value is provided as a Pixel object:
h_init = Pixel(0, 0, 0)

# Now, we can perform the reduction:

# compute temp storage:
reducer = cudax.reduce_into(d_rgb, d_out, max_g_value, h_init)
temp_storage_bytes = reducer(None, d_rgb, d_out, len(d_rgb), h_init)

# do the reduction:
d_temp_storage = cp.zeros(temp_storage_bytes, dtype=np.uint8)
_ = reducer(d_temp_storage, d_rgb, d_out, len(d_rgb), h_init)

# results:
print()
print("Input RGB values:")
print("-----------------")
print(d_rgb.get())
print()
print("Value with largest g component:")
print("-------------------------------")
print(d_out.get())
print()
