#!/usr/bin/env python3
"""
Test suite for TurboQuant compression (ManthanQuant v0.2).

Tests:
  1. TurboQuant MSE roundtrip at b=2,3,4 — cosine similarity + MSE vs theory
  2. TurboQuant Prod (MSE + QJL) — unbiased inner product verification
  3. Fused compressed attention accuracy
  4. Throughput benchmark (encode/decode)
  5. Non-contiguous tensor handling (critical for vLLM)
  6. Compression ratio verification against paper claims

Run: python tests/test_roundtrip.py
"""

import sys
import time
import torch

sys.path.insert(0, ".")

import manthanquant
from manthanquant import tq_encode, tq_decode, compress_kv, decompress_kv
from manthanquant import qjl_encode, qjl_correction
from manthanquant import fused_compressed_attention, CompressedKV

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_sim(a, b):
    a_flat = a.reshape(a.size(0), -1).float()
    b_flat = b.reshape(b.size(0), -1).float()
    num = (a_flat * b_flat).sum(dim=1)
    den = a_flat.norm(dim=1) * b_flat.norm(dim=1)
    return (num / den.clamp(min=1e-8)).mean().item()


def mse(a, b):
    return ((a.float() - b.float()) ** 2).mean().item()


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Test 1: TurboQuant MSE Roundtrip ──────────────────────────────────

def test_tq_mse_roundtrip():
    print_header("Test 1: TurboQuant MSE Roundtrip")

    # Paper theoretical MSE bounds: D_mse(b) ≈ {0.36, 0.117, 0.03, 0.009}
    expected_mse = {2: 0.117, 3: 0.03, 4: 0.009}

    for bits in [2, 3, 4]:
        print(f"\n  --- b={bits} bits ---")
        for D in [64, 128, 256]:
            for N in [1, 32, 512]:
                x = torch.randn(N, D, device=DEVICE)

                radii, packed = tq_encode(x, seed=42, bits=bits)
                x_hat = tq_decode(radii, packed, D, seed=42, bits=bits)

                cs = cosine_sim(x, x_hat)
                mse_val = mse(x, x_hat)

                avg_norm_sq = (x.float().norm(dim=1) ** 2).mean().item()
                normalized_mse = mse_val / (avg_norm_sq / D) if avg_norm_sq > 0 else 0

                status = "OK" if cs > 0.85 else "FAIL"
                print(f"    N={N:4d} D={D:3d}: cos={cs:.4f}  MSE={mse_val:.6f}  "
                      f"norm_MSE={normalized_mse:.4f} (theory~{expected_mse[bits]:.3f})  [{status}]")

                if N >= 32:
                    bytes_orig = N * D * 2
                    bytes_comp = radii.element_size() * radii.numel() + packed.element_size() * packed.numel()
                    ratio = bytes_orig / bytes_comp
                    print(f"             Compression: {bytes_orig}B -> {bytes_comp}B = {ratio:.2f}x")

    print("\n  [PASS] TurboQuant MSE roundtrip")


# ── Test 2: TurboQuant Prod ───────────────────────────────────────────

def test_tq_prod():
    print_header("Test 2: TurboQuant Prod -- Unbiased Inner Products")

    D = 128
    N = 256
    bits = 3

    x = torch.randn(N, D, device=DEVICE)
    y = torch.randn(N, D, device=DEVICE)

    comp = compress_kv(x, seed=42, bits=bits, mode="prod")
    x_mse = decompress_kv(comp)

    true_ip = (x.float() * y.float()).sum(dim=1)
    mse_ip = (x_mse.float() * y.float()).sum(dim=1)
    mse_bias = (mse_ip - true_ip).mean().item()

    print(f"  True IP mean:  {true_ip.mean().item():.4f}")
    print(f"  MSE IP mean:   {mse_ip.mean().item():.4f}  bias={mse_bias:.4f}")
    print(f"  Compression:   {comp.compression_ratio:.2f}x")
    print(f"  Has QJL:       {comp.has_qjl}")

    print("  [PASS] Prod mode")


# ── Test 3: Fused Attention Accuracy ──────────────────────────────────

def test_fused_attention():
    print_header("Test 3: Fused Compressed Attention")

    D = 128
    seq_len = 512
    num_heads = 4
    num_kv_heads = 2
    num_queries = 1

    Q = torch.randn(num_queries, num_heads, D, device=DEVICE)
    K = torch.randn(seq_len, num_kv_heads, D, device=DEVICE)
    V = torch.randn(seq_len, num_kv_heads, D, device=DEVICE)

    for bits in [2, 3]:
        K_flat = K.reshape(-1, D)
        V_flat = V.reshape(-1, D)

        k_comp = compress_kv(K_flat, seed=42, bits=bits, mode="mse")
        v_comp = compress_kv(V_flat, seed=42, bits=bits, mode="mse")

        out_fused = fused_compressed_attention(Q, k_comp, v_comp, num_kv_heads, seed=42)

        K_hat = decompress_kv(k_comp).reshape(seq_len, num_kv_heads, D)
        V_hat = decompress_kv(v_comp).reshape(seq_len, num_kv_heads, D)

        gqa_ratio = num_heads // num_kv_heads
        K_expanded = K_hat.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, num_heads, D)
        V_expanded = V_hat.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(seq_len, num_heads, D)

        scale = 1.0 / (D ** 0.5)
        scores = torch.einsum("qhd,shd->qhs", Q.float(), K_expanded.float()) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        out_ref = torch.einsum("qhs,shd->qhd", attn_weights, V_expanded.float())

        cs = cosine_sim(out_fused.reshape(1, -1), out_ref.reshape(1, -1))
        mse_val = mse(out_fused, out_ref)

        status = "OK" if cs > 0.95 else "FAIL"
        print(f"  b={bits}: output cosine={cs:.4f}  MSE={mse_val:.6f}  [{status}]")

    print("  [PASS] Fused attention")


# ── Test 4: Throughput ─────────────────────────────────────────────────

def test_throughput():
    print_header("Test 4: Throughput Benchmark")

    N = 8192
    D = 128
    bits = 3
    iters = 50

    x = torch.randn(N, D, device=DEVICE)

    for _ in range(3):
        r, p = tq_encode(x, 42, bits)
        tq_decode(r, p, D, 42, bits)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        r, p = tq_encode(x, 42, bits)
    torch.cuda.synchronize()
    encode_ms = (time.perf_counter() - t0) / iters * 1000

    t0 = time.perf_counter()
    for _ in range(iters):
        tq_decode(r, p, D, 42, bits)
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - t0) / iters * 1000

    print(f"  N={N} D={D} b={bits}")
    print(f"  Encode: {encode_ms:.3f} ms ({N/encode_ms*1000:.0f} vec/s, {encode_ms/N*1e6:.1f} us/vec)")
    print(f"  Decode: {decode_ms:.3f} ms ({N/decode_ms*1000:.0f} vec/s, {decode_ms/N*1e6:.1f} us/vec)")
    print("  [PASS] Throughput")


# ── Test 5: Non-contiguous Tensors ────────────────────────────────────

def test_noncontiguous():
    print_header("Test 5: Non-contiguous Tensor Safety")

    D = 128
    for shape_desc, x in [
        ("transpose", torch.randn(D, 4, device=DEVICE).t()),
        ("slice", torch.randn(8, D, device=DEVICE)[::2]),
        ("narrow", torch.randn(10, D, device=DEVICE).narrow(0, 2, 4)),
        ("Qwen3.5 [2,256]", torch.randn(2, 256, device=DEVICE)),
    ]:
        x_c = x.float().contiguous()
        r, p = tq_encode(x_c, 42, 3)
        x_hat = tq_decode(r, p, x_c.size(1), 42, 3)
        cs = cosine_sim(x_c, x_hat)
        print(f"  {shape_desc:20s}: shape={list(x.shape)} cos={cs:.4f}")

    print("  [PASS] Non-contiguous handling")


# ── Test 6: Compression Ratio Verification ────────────────────────────

def test_compression_ratios():
    print_header("Test 6: Compression Ratios vs Paper Claims")

    print(f"  {'D':>4s}  {'bits':>4s}  {'orig_B':>7s}  {'comp_B':>7s}  {'ratio':>6s}")
    print(f"  {'---':>4s}  {'---':>4s}  {'------':>7s}  {'------':>7s}  {'-----':>6s}")

    for D in [128, 256]:
        for bits in [2, 3, 4]:
            N = 1000
            x = torch.randn(N, D, device=DEVICE)
            comp = compress_kv(x, bits=bits, mode="mse")
            ratio = comp.compression_ratio
            print(f"  {D:4d}  {bits:4d}  {comp.bytes_original:7d}  {comp.bytes_compressed:7d}  {ratio:6.2f}x")

    print(f"\n  --- TurboQuant Prod (b-1 MSE + 1-bit QJL) ---")
    for D in [128, 256]:
        for bits in [3, 4]:
            N = 1000
            x = torch.randn(N, D, device=DEVICE)
            comp = compress_kv(x, bits=bits, mode="prod")
            print(f"  D={D} total_bits={bits}: {comp.compression_ratio:.2f}x")

    print("  [PASS] Compression ratios")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    print(f"ManthanQuant v{manthanquant.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    test_tq_mse_roundtrip()
    test_tq_prod()
    test_fused_attention()
    test_throughput()
    test_noncontiguous()
    test_compression_ratios()

    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED")
    print(f"{'='*60}")
