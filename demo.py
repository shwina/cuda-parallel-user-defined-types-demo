import numpy as np
import cupy as cp

import cuda.parallel.experimental as cudax
from gpudataclass import gpudataclass



@gpudataclass
class Pixel:
    r: np.dtype("int32")
    g: np.dtype("int32")
    b: np.dtype("int32")
    
dtype = np.dtype([("r", "int32"), ("g", "int32"), ("b", "int32")])

# Create a CuPy array of 10 RGB values (10 rows, 3 columns), and view it
# as an array of type `dtype`
rgb_values = cp.random.randint(0, 256, (10, 3), dtype=cp.int32).view(dtype)

d_rgb = rgb_values
d_out = cp.zeros(1, dtype)
h_init = Pixel(0, 0, 0)

# define comparator:
def max_g_value(x, y):
    return x if x.g > y.g else y

# compute temp storage:
#value_type = numba.from_dtype(dtype)
reducer = cudax.reduce_into(d_rgb, d_out, max_g_value, h_init)
temp_storage_bytes = reducer(None, d_rgb, d_out, len(d_rgb), h_init)

# do the reduction:
d_temp_storage = cp.zeros(temp_storage_bytes, dtype=np.uint8)
_ = reducer(d_temp_storage, d_rgb, d_out, len(d_rgb), h_init)

# results:
print(d_rgb.get())
print(d_out.get())
