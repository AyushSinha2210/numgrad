# numgrad/memory.py
import time
from typing import Dict, Any, Tuple
import numpy as np


def inspect_memory(arr: np.ndarray) -> Dict[str, Any]:
    """Extracts low-level C buffer metadata and stride geometry."""
    return {
        "memory_address": hex(arr.ctypes.data),
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "itemsize_bytes": arr.itemsize,
        "strides_bytes": arr.strides,
        "nbytes_total": arr.nbytes,
        "c_contiguous": arr.flags["C_CONTIGUOUS"],
        "f_contiguous": arr.flags["F_CONTIGUOUS"],
        "owns_data": arr.flags["OWNDATA"],
    }


def benchmark_contiguity(size: int = 8000) -> Tuple[float, float, bool]:
    """
    Creates a C-contiguous matrix and a non-contiguous transposed view.
    Proves they share the exact same physical RAM address, then benchmarks
    row-wise summation speed (L1/L2 cache hit vs cache miss).
    """
    c_array = np.ones((size, size), dtype=np.float64)
    transposed_view = c_array.T  # Strides are flipped; C_CONTIGUOUS is now False

    # Verify zero-copy view (same starting physical address)
    shares_memory = (c_array.ctypes.data == transposed_view.ctypes.data)

    # Warmup
    _ = np.sum(c_array, axis=1)
    _ = np.sum(transposed_view, axis=1)

    # Benchmark C-Contiguous (Sequential memory reads = Cache Hits)
    t0 = time.perf_counter()
    for _ in range(10):
        _ = np.sum(c_array, axis=1)
    c_time = (time.perf_counter() - t0) / 10

    # Benchmark Non-Contiguous View (Strided jumps = Cache Misses)
    t1 = time.perf_counter()
    for _ in range(10):
        _ = np.sum(transposed_view, axis=1)
    strided_time = (time.perf_counter() - t1) / 10

    return c_time, strided_time, shares_memory