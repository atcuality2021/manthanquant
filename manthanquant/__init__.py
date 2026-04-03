"""ManthanQuant — 3-bit Lloyd-Max KV cache compression for vLLM.

Supports two backends:
  - GB10 (ARM unified memory): Pure numpy on CPU cores — zero-copy, no CUDA conflicts
  - x86 (discrete GPU): CUDA kernels on GPU — no PCIe transfer, 10x faster compression

Auto-selects backend based on hardware.
"""
__version__ = "0.4.0"

import platform

IS_ARM = platform.machine() in ("aarch64", "arm64")
IS_X86 = platform.machine() in ("x86_64", "AMD64")

# GB10 / ARM path
try:
    from manthanquant.cpu_quantize import tq_encode_numpy, tq_decode_numpy
except ImportError:
    pass

# x86 / discrete GPU path
try:
    from manthanquant.x86_quantize import encode, decode, HAS_CUDA_EXT
except ImportError:
    HAS_CUDA_EXT = False

# Unified API — auto-selects best backend
try:
    from manthanquant.x86_quantize import encode as compress, decode as decompress
except ImportError:
    # x86_quantize unavailable (e.g. no torch) — fall back to numpy
    try:
        from manthanquant.cpu_quantize import tq_encode_numpy as compress, tq_decode_numpy as decompress
    except ImportError:
        pass
