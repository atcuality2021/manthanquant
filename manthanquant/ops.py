"""
High-level Python API for ManthanQuant / TurboQuant compression ops.

All heavy lifting is done in CUDA kernels (_C extension).
This module provides a clean interface + the combined compress/decompress flow.

TurboQuant modes:
  - tq_mse (b bits):  Optimal MSE quantizer, b bits per coordinate
  - tq_prod (b bits):  (b-1)-bit MSE + 1-bit QJL = unbiased inner product estimator

For KV cache in attention, tq_prod is preferred (unbiased attention scores).
"""

import torch
from dataclasses import dataclass
from typing import Optional

import manthanquant._C as _C


# ── Compression output container ──────────────────────────────────────────

@dataclass
class CompressedKV:
    """Compressed KV cache entry for one layer/head.

    TurboQuant format:
      - radii:  L2 norms of original vectors
      - packed: b-bit quantization indices (Lloyd-Max centroids)
      - qjl_signs:  1-bit QJL error correction signs (optional, for tq_prod mode)
      - qjl_gamma:  residual norms for QJL scaling (optional)
    """
    radii: torch.Tensor           # [N] float32 — vector magnitudes
    packed: torch.Tensor           # [N, words] int32 — b-bit packed indices
    dim: int                       # Original vector dimension D
    bits: int                      # Quantization bit width (2, 3, or 4)
    seed: int                      # Seed for WHT rotation
    # QJL correction (for tq_prod mode)
    qjl_signs: Optional[torch.Tensor] = None   # [N, sign_words] int32
    qjl_gamma: Optional[torch.Tensor] = None   # [N] float32 — ||residual||
    qjl_seed: int = 137
    qjl_m: int = -1

    @property
    def num_vectors(self) -> int:
        return self.radii.size(0)

    @property
    def bytes_compressed(self) -> int:
        total = (
            self.radii.element_size() * self.radii.numel() +
            self.packed.element_size() * self.packed.numel()
        )
        if self.qjl_signs is not None:
            total += self.qjl_signs.element_size() * self.qjl_signs.numel()
        if self.qjl_gamma is not None:
            total += self.qjl_gamma.element_size() * self.qjl_gamma.numel()
        return total

    @property
    def bytes_original(self) -> int:
        return self.num_vectors * self.dim * 2  # bf16 = 2 bytes

    @property
    def compression_ratio(self) -> float:
        if self.bytes_compressed == 0:
            return 0.0
        return self.bytes_original / self.bytes_compressed

    @property
    def has_qjl(self) -> bool:
        return self.qjl_signs is not None


# ── TurboQuant low-level ops ─────────────────────────────────────────────

def tq_encode(
    vectors: torch.Tensor,
    seed: int = 42,
    bits: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    TurboQuant MSE encode: [N, D] → (radii [N], packed [N, words])

    D must be a power of 2 (64, 128, 256 typical for attention heads).
    """
    return _C.tq_encode(vectors, seed, bits)


def tq_decode(
    radii: torch.Tensor,
    packed: torch.Tensor,
    D: int,
    seed: int = 42,
    bits: int = 3,
) -> torch.Tensor:
    """
    TurboQuant MSE decode: (radii, packed) → [N, D] float32
    """
    return _C.tq_decode(radii, packed, D, seed, bits)


def qjl_encode(
    errors: torch.Tensor,
    M: int = -1,
    seed: int = 137,
) -> torch.Tensor:
    """QJL encode: error vectors [N, D] → packed sign bits [N, sign_words]"""
    return _C.qjl_encode(errors, M, seed)


def qjl_correction(
    queries: torch.Tensor,
    key_signs: torch.Tensor,
    D: int,
    M: int = -1,
    seed: int = 137,
) -> torch.Tensor:
    """QJL correction: compute additive bias correction for attention scores."""
    return _C.qjl_correction(queries, key_signs, D, M, seed)


# ── High-level compress/decompress ──────────────────────────────────────

def compress_kv(
    vectors: torch.Tensor,
    seed: int = 42,
    bits: int = 3,
    mode: str = "mse",
    qjl_seed: int = 137,
    qjl_m: int = -1,
) -> CompressedKV:
    """
    Full TurboQuant compression pipeline.

    Args:
        vectors: [N, D] tensor (key or value vectors for one head)
        seed: seed for WHT random rotation
        bits: total bit budget per coordinate (2, 3, or 4)
        mode: "mse" for TurboQuant_mse, "prod" for TurboQuant_prod
        qjl_seed: seed for QJL Rademacher matrix (prod mode only)
        qjl_m: QJL projection dimension (-1 = D)

    Returns:
        CompressedKV with all components.

    Mode details:
        mse:  b bits for MSE-optimal quantizer. Best reconstruction quality.
        prod: (b-1) bits MSE + 1 bit QJL on residual. Unbiased inner products.
    """
    D = vectors.size(1)
    if qjl_m <= 0:
        qjl_m = D

    if mode == "prod":
        # TurboQuant_prod: (b-1)-bit MSE + 1-bit QJL
        mse_bits = bits - 1
        assert mse_bits >= 1, f"prod mode needs bits >= 2, got {bits}"

        # Stage 1: MSE encode at (b-1) bits
        radii, packed = tq_encode(vectors, seed, mse_bits)

        # Stage 2: compute residual
        reconstructed = tq_decode(radii, packed, D, seed, mse_bits)
        residual = vectors.float() - reconstructed

        # Compute residual norms for scaling
        qjl_gamma = torch.norm(residual, dim=1)

        # Stage 3: QJL encode the residual
        qjl_signs = qjl_encode(residual, qjl_m, qjl_seed)

        return CompressedKV(
            radii=radii, packed=packed, dim=D, bits=mse_bits,
            seed=seed,
            qjl_signs=qjl_signs, qjl_gamma=qjl_gamma,
            qjl_seed=qjl_seed, qjl_m=qjl_m,
        )
    else:
        # TurboQuant_mse: b bits MSE-only
        radii, packed = tq_encode(vectors, seed, bits)

        return CompressedKV(
            radii=radii, packed=packed, dim=D, bits=bits,
            seed=seed,
        )


def decompress_kv(compressed: CompressedKV) -> torch.Tensor:
    """
    Decompress back to float32 vectors (MSE decode only).

    For TurboQuant_prod with full accuracy, use the fused attention kernel
    which applies QJL correction inline during attention computation.
    """
    return tq_decode(
        compressed.radii, compressed.packed,
        compressed.dim, compressed.seed, compressed.bits,
    )


# ── Fused compressed attention ──────────────────────────────────────────

def fused_compressed_attention(
    queries: torch.Tensor,
    k_compressed: CompressedKV,
    v_compressed: CompressedKV,
    num_kv_heads: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Fused attention: Q + compressed KV → output.

    Decompresses K, V on-the-fly inside the kernel. No intermediate buffer.

    Args:
        queries: [num_queries, num_heads, head_dim]
        k_compressed: CompressedKV for keys
        v_compressed: CompressedKV for values
        num_kv_heads: number of KV heads (for GQA)
        seed: WHT seed (must match compression)

    Returns:
        [num_queries, num_heads, head_dim] attention output
    """
    total_vecs = k_compressed.num_vectors
    seq_len = total_vecs // num_kv_heads
    bits = k_compressed.bits

    k_radii = k_compressed.radii.view(seq_len, num_kv_heads)
    k_packed = k_compressed.packed.view(seq_len, num_kv_heads, -1)
    v_radii = v_compressed.radii.view(seq_len, num_kv_heads)
    v_packed = v_compressed.packed.view(seq_len, num_kv_heads, -1)

    return _C.fused_attention(
        queries, k_radii, k_packed, v_radii, v_packed,
        num_kv_heads, seed, bits,
    )


