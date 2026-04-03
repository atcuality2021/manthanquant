"""
x86_quantize.py — CUDA-accelerated Lloyd-Max 3-bit quantization for x86 discrete GPUs.

Uses custom CUDA kernels via manthanquant._C extension.
On x86 with discrete GPUs (RTX 4070, RTX 6000, A100, etc.), data lives in VRAM
and compression runs on GPU — no PCIe transfers needed.

Compression format (per vector of dim D):
  - radius: float32 (4 bytes) — L2 norm
  - packed: int32 array of ceil(D*bits/32) words — bit-packed centroid indices

For D=256, bits=3:
  Original bf16:  256 × 2 = 512 bytes
  Compressed:     4 + ceil(256×3/32)*4 = 4 + 96 = 100 bytes
  Ratio:          512 / 100 = 5.12x

This module provides the same API as cpu_quantize.py but runs on GPU.
"""

import torch
from typing import Tuple, Optional

# Try loading CUDA extension
try:
    import manthanquant._C as _C
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False

# Lloyd-Max optimal centroids for 3-bit (8 levels), unit Gaussian N(0,1).
CENTROIDS_3BIT = torch.tensor([
    -2.151946, -1.343910, -0.756006, -0.245094,
     0.245094,  0.756006,  1.343910,  2.151946
], dtype=torch.float32)

BOUNDARIES_3BIT = torch.tensor([
    -1.747928, -1.049958, -0.500550, 0.000000,
     0.500550,  1.049958,  1.747928
], dtype=torch.float32)


def tq_encode_cuda(
    vectors: torch.Tensor,
    bits: int = 3,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode vectors using Lloyd-Max quantization on GPU via CUDA kernels.

    Args:
        vectors: [N, D] bf16/fp16/fp32 tensor on CUDA device.
        bits: quantization bits (2, 3, or 4).
        seed: random seed for Walsh-Hadamard rotation.

    Returns:
        radii:  [N] float32 tensor on same device — L2 norms
        packed: [N, words] int32 tensor on same device — bit-packed indices
    """
    if not HAS_CUDA_EXT:
        raise RuntimeError(
            "manthanquant._C not available. Build with: "
            "cd manthanquant && pip install -e ."
        )

    assert vectors.is_cuda, "Input must be on CUDA device"
    assert vectors.ndim == 2, f"Expected [N, D], got {vectors.shape}"

    N, D = vectors.shape
    vectors_f32 = vectors.float()  # Ensure float32

    # Use CUDA kernel for encoding
    radii, packed = _C.tq_encode(vectors_f32, seed, bits)
    return radii, packed


def tq_decode_cuda(
    radii: torch.Tensor,
    packed: torch.Tensor,
    dim: int,
    bits: int = 3,
    seed: int = 42,
) -> torch.Tensor:
    """Decode compressed vectors on GPU via CUDA kernels.

    Args:
        radii:  [N] float32 tensor — L2 norms
        packed: [N, words] int32 tensor — bit-packed indices
        dim: original vector dimension D
        bits: quantization bits
        seed: same seed used during encoding

    Returns:
        vectors: [N, D] float32 tensor — reconstructed vectors
    """
    if not HAS_CUDA_EXT:
        raise RuntimeError("manthanquant._C not available")

    return _C.tq_decode(radii, packed, dim, seed, bits)


def _make_pack_tables(D: int, bits: int, device: torch.device):
    """Pre-compute vectorized pack/unpack index tables for given D and bits."""
    pos = torch.arange(D, device=device)
    bit_pos = pos * bits
    word_idx = bit_pos // 32
    bit_offset = bit_pos % 32
    return word_idx, bit_offset


# Cache tables per (D, bits, device) to avoid recomputation
_pack_tables_cache: dict = {}


def _get_pack_tables(D: int, bits: int, device: torch.device):
    key = (D, bits, str(device))
    if key not in _pack_tables_cache:
        _pack_tables_cache[key] = _make_pack_tables(D, bits, device)
    return _pack_tables_cache[key]


def tq_encode_torch(
    vectors: torch.Tensor,
    bits: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch encoder — vectorized bit-packing, no Python for-loops.

    Works on any device (CPU/GPU). On GPU, all operations are vectorized
    torch ops — no per-element Python loops.
    """
    assert vectors.ndim == 2
    N, D = vectors.shape
    vectors_f32 = vectors.float()

    # 1. Compute L2 norms
    radii = torch.norm(vectors_f32, dim=1)  # [N]

    # 2. Normalize and scale to N(0,1) space
    safe_radii = radii.clamp(min=1e-8)
    normalized = vectors_f32 / safe_radii.unsqueeze(1)  # [N, D]
    scaled = normalized * (D ** 0.5)  # Map to N(0,1)

    # 3. Quantize: find nearest centroid
    boundaries = BOUNDARIES_3BIT.to(vectors.device)
    indices = torch.searchsorted(boundaries, scaled.reshape(-1))  # [N*D]
    indices = indices.reshape(N, D).to(torch.int32)

    # 4. Vectorized bit-pack into int32 words
    # Group indices by target word and scatter-add shifted values
    num_words = (D * bits + 31) // 32
    word_idx, bit_offset = _get_pack_tables(D, bits, vectors.device)

    # Shift each index to its position: [N, D]
    shifted = (indices & ((1 << bits) - 1)) << bit_offset.unsqueeze(0)  # [N, D]

    # Scatter-add into packed words using scatter_add
    # Expand word_idx to [N, D]
    word_idx_exp = word_idx.unsqueeze(0).expand(N, -1).long()
    packed = torch.zeros(N, num_words, dtype=torch.int32, device=vectors.device)
    # scatter_add doesn't support int32, so use int64 and cast back
    packed_i64 = torch.zeros(N, num_words, dtype=torch.int64, device=vectors.device)
    packed_i64.scatter_add_(1, word_idx_exp, shifted.to(torch.int64))
    packed = packed_i64.to(torch.int32)

    return radii, packed


def tq_decode_torch(
    radii: torch.Tensor,
    packed: torch.Tensor,
    dim: int,
    bits: int = 3,
) -> torch.Tensor:
    """Pure PyTorch decoder — vectorized bit-unpacking."""
    N = radii.shape[0]
    centroids = CENTROIDS_3BIT.to(packed.device)
    mask = (1 << bits) - 1

    word_idx, bit_offset = _get_pack_tables(dim, bits, packed.device)

    # Gather the relevant words for each position: [N, D]
    word_idx_exp = word_idx.unsqueeze(0).expand(N, -1).long()
    words = torch.gather(packed, 1, word_idx_exp)  # [N, D]

    # Shift right and mask to extract indices
    indices = (words >> bit_offset.unsqueeze(0)) & mask  # [N, D]

    # Look up centroids
    scaled = centroids[indices.long()]  # [N, D]

    # Undo scaling and restore magnitude
    safe_radii = radii.clamp(min=1e-8)
    reconstructed = (scaled / (dim ** 0.5)) * safe_radii.unsqueeze(1)

    return reconstructed


# ── Convenience: auto-select best backend ────────────────────────────────

def encode(vectors: torch.Tensor, bits: int = 3, seed: int = 42):
    """Encode vectors using the best available backend.

    Returns (radii, packed) tensors on the same device as input.
    """
    if vectors.is_cuda and HAS_CUDA_EXT:
        return tq_encode_cuda(vectors, bits, seed)
    elif vectors.is_cuda:
        return tq_encode_torch(vectors, bits)
    else:
        # CPU fallback — use numpy path
        import numpy as np
        from .cpu_quantize import tq_encode_numpy
        v_np = vectors.float().numpy()
        radii_np, packed_np = tq_encode_numpy(v_np, bits)
        return torch.from_numpy(radii_np), torch.from_numpy(packed_np)


def decode(radii, packed, dim: int, bits: int = 3, seed: int = 42):
    """Decode compressed vectors using the best available backend."""
    if radii.is_cuda and HAS_CUDA_EXT:
        return tq_decode_cuda(radii, packed, dim, bits, seed)
    elif radii.is_cuda:
        return tq_decode_torch(radii, packed, dim, bits)
    else:
        from .cpu_quantize import tq_decode_numpy
        r_np = radii.numpy()
        p_np = packed.numpy()
        return torch.from_numpy(tq_decode_numpy(r_np, p_np, dim, bits))
