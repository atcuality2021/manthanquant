"""ManthanQuant — 3-bit Lloyd-Max KV cache compression for vLLM."""
__version__ = "0.3.0"

try:
    from manthanquant.cpu_quantize import tq_encode_numpy, tq_decode_numpy
except ImportError:
    pass
