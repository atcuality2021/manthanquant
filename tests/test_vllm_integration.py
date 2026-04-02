#!/usr/bin/env python3
"""
test_vllm_integration.py — Simulate vLLM call patterns to verify ManthanQuant hooks.

Tests WITHOUT a real model:
  1. Layer tracking: id(self) works for multiple attention instances
  2. Hybrid architecture: Mamba layers skipped, attention layers tracked
  3. Chunked prefill: multiple chunks per request compress correctly
  4. Decode: fused attention reads from correct per-layer shadow cache
  5. Multi-request: cache cleared between requests
  6. No CUDA conflicts: compression happens AFTER forward, not during

Run: python tests/test_vllm_integration.py
"""

import sys
import torch
import time

sys.path.insert(0, ".")

import manthanquant._C as _C
from manthanquant.vllm_patch import (
    _patched_kv_hook,
    _patched_forward_hook,
    _patched_forward_post_hook,
    _flush_pending_kv,
    _get_layer_name,
    _is_first_layer,
    _shadow_cache,
    _pending_kv,
    _instance_to_name,
    _instance_order,
    _stats,
    clear_cache,
    LayerCache,
    SEED, BITS,
)

DEVICE = "cuda"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


class FakeAttnMetadata:
    """Simulates vLLM's attn_metadata."""
    def __init__(self, num_actual_tokens, max_query_len):
        self.num_actual_tokens = num_actual_tokens
        self.max_query_len = max_query_len


class FakeFlashAttentionImpl:
    """Simulates one FlashAttentionImpl instance per attention layer."""
    def __init__(self, num_kv_heads=4, head_dim=128):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


class FakeMambaLayer:
    """Simulates a Mamba/SSM layer — never calls FlashAttention hooks."""
    pass


def reset_global_state():
    """Reset all vllm_patch global state for a clean test."""
    import manthanquant.vllm_patch as patch
    patch._shadow_cache.clear()
    patch._pending_kv.clear()
    patch._instance_to_name.clear()
    patch._instance_order.clear()
    patch._first_instance_id = 0
    patch._request_count = 0
    patch._warmup_done = True  # Skip warmup logic in tests
    patch._last_was_decode = False
    patch._stats = {
        "decode_calls": 0, "prefill_calls": 0,
        "decode_fused": 0, "decode_fallback": 0,
        "compressed_bytes": 0, "layers_discovered": 0,
    }


# ── Test 1: Layer tracking with id(self) ─────────────────────────────

def test_layer_tracking():
    print_header("Test 1: Layer Tracking (id-based)")
    reset_global_state()

    layers = [FakeFlashAttentionImpl() for _ in range(8)]

    names = [_get_layer_name(l) for l in layers]
    print(f"  8 layers: {names}")

    assert len(set(names)) == 8, f"Expected 8 unique names, got {len(set(names))}"

    # Same instances should return same names
    names2 = [_get_layer_name(l) for l in layers]
    assert names == names2, "Names not stable across calls"

    assert _is_first_layer(layers[0]), "First layer not detected"
    assert not _is_first_layer(layers[1]), "Second layer wrongly detected as first"

    print("  [PASS] Layer tracking works")


# ── Test 2: Hybrid architecture (Mamba + attention) ──────────────────

def test_hybrid_architecture():
    print_header("Test 2: Hybrid Architecture (Mamba + Attention)")
    reset_global_state()

    # Simulate Qwen3.5-35B architecture: some layers are Mamba, some are attention
    architecture = []
    for i in range(10):
        if i % 3 == 0:
            architecture.append(("mamba", FakeMambaLayer()))
        else:
            architecture.append(("attn", FakeFlashAttentionImpl()))

    # Only attention layers register
    attn_names = []
    for kind, layer in architecture:
        if kind == "attn":
            name = _get_layer_name(layer)
            attn_names.append(name)

    num_attn = sum(1 for k, _ in architecture if k == "attn")
    assert len(attn_names) == num_attn
    assert len(set(attn_names)) == num_attn, "Duplicate attention layer names"

    print(f"  Architecture: {[k for k, _ in architecture]}")
    print(f"  Attention layers tracked: {attn_names}")
    print("  [PASS] Hybrid architecture supported")


# ── Test 3: KV hook queues without CUDA kernels ──────────────────────

def test_kv_hook_deferred():
    print_header("Test 3: KV Hook Deferred Compression")
    reset_global_state()

    D = 128
    KH = 4
    attn = FakeFlashAttentionImpl(num_kv_heads=KH, head_dim=D)
    layer_name = _get_layer_name(attn)

    # Simulate do_kv_cache_update call
    num_tokens = 32
    key = torch.randn(num_tokens, KH, D, device=DEVICE)
    value = torch.randn(num_tokens, KH, D, device=DEVICE)
    slot_mapping = torch.arange(num_tokens, device=DEVICE)

    # kv_hook should only queue, NOT compress
    _patched_kv_hook(attn, None, key, value, None, slot_mapping)

    assert layer_name in _pending_kv, f"Expected {layer_name} in pending_kv"
    assert layer_name not in _shadow_cache or _shadow_cache[layer_name].seq_len == 0, \
        "Shadow cache should be empty (compression is deferred)"

    pending_k, pending_v = _pending_kv[layer_name]
    assert pending_k.shape == (num_tokens, KH, D), f"Wrong shape: {pending_k.shape}"

    print(f"  Queued {num_tokens} tokens for {layer_name}")
    print(f"  Pending KV count: {len(_pending_kv)}")
    print(f"  Shadow cache seq_len: {_shadow_cache.get(layer_name, LayerCache()).seq_len}")
    print("  [PASS] KV hook defers compression")


# ── Test 4: Post-forward flush compresses correctly ──────────────────

def test_post_forward_flush():
    print_header("Test 4: Post-Forward Flush")
    # Continuing from test 3 state (pending_kv has data)

    attn = list(_instance_to_name.keys())
    if not attn:
        print("  [SKIP] No layers registered")
        return

    layer_name = list(_pending_kv.keys())[0]
    assert layer_name in _pending_kv

    # Flush — this should compress the queued KV
    _patched_forward_post_hook(None, layer_name)

    assert layer_name not in _pending_kv, "Pending should be cleared after flush"
    assert layer_name in _shadow_cache, "Shadow cache should have data"
    assert _shadow_cache[layer_name].seq_len > 0, "Shadow cache should have tokens"

    print(f"  {layer_name} shadow cache: {_shadow_cache[layer_name].seq_len} tokens")
    print(f"  Compressed bytes: {_shadow_cache[layer_name].memory_bytes()}")
    print("  [PASS] Post-forward flush works")


# ── Test 5: Full prefill + decode cycle ──────────────────────────────

def test_full_cycle():
    print_header("Test 5: Full Prefill + Decode Cycle")
    reset_global_state()

    D = 128
    KH = 4
    NH = 16  # attention heads (GQA 16:4 = 4x)
    NUM_ATTN_LAYERS = 4
    SEQ_LEN = 64

    # Create attention layer instances
    attn_layers = [FakeFlashAttentionImpl(num_kv_heads=KH, head_dim=D)
                   for _ in range(NUM_ATTN_LAYERS)]

    # Register all layers
    for a in attn_layers:
        _get_layer_name(a)

    # ── PREFILL phase: simulate chunked prefill ──
    chunk_size = 32
    for chunk_start in range(0, SEQ_LEN, chunk_size):
        chunk_end = min(chunk_start + chunk_size, SEQ_LEN)
        num_tokens = chunk_end - chunk_start

        for attn in attn_layers:
            layer_name = _get_layer_name(attn)

            # 1. KV hook (queue)
            key = torch.randn(num_tokens, KH, D, device=DEVICE)
            value = torch.randn(num_tokens, KH, D, device=DEVICE)
            slot_mapping = torch.arange(num_tokens, device=DEVICE)
            _patched_kv_hook(attn, None, key, value, None, slot_mapping)

            # 2. Forward hook (prefill → returns None, FlashAttention handles)
            query = torch.randn(num_tokens, NH, D, device=DEVICE)
            meta = FakeAttnMetadata(num_actual_tokens=num_tokens, max_query_len=num_tokens)
            result = _patched_forward_hook(
                attn, None, query, key, value, None,
                meta, None, None, None
            )
            assert result is None, "Prefill should return None (let FlashAttention handle)"

            # 3. Post-forward flush (compress queued KV)
            _patched_forward_post_hook(attn, layer_name)

    # Verify shadow cache
    for attn in attn_layers:
        name = _get_layer_name(attn)
        cache = _shadow_cache.get(name)
        assert cache is not None, f"No cache for {name}"
        assert cache.seq_len == SEQ_LEN, f"{name}: expected {SEQ_LEN} tokens, got {cache.seq_len}"

    print(f"  Prefill done: {NUM_ATTN_LAYERS} layers × {SEQ_LEN} tokens")
    for a in attn_layers:
        n = _get_layer_name(a)
        print(f"    {n}: {_shadow_cache[n].seq_len} tokens, {_shadow_cache[n].memory_bytes()} bytes")

    # ── DECODE phase: fused compressed attention ──
    decode_query = torch.randn(1, NH, D, device=DEVICE)
    decode_meta = FakeAttnMetadata(num_actual_tokens=1, max_query_len=1)

    decode_results = []
    for attn in attn_layers:
        result = _patched_forward_hook(
            attn, None, decode_query, None, None, None,
            decode_meta, None, None, None
        )
        if result is not None:
            decode_results.append((_get_layer_name(attn), result))

    print(f"\n  Decode results: {len(decode_results)}/{NUM_ATTN_LAYERS} layers used fused attention")
    for name, res in decode_results:
        print(f"    {name}: output shape={list(res.shape)}")

    assert len(decode_results) == NUM_ATTN_LAYERS, \
        f"Expected all {NUM_ATTN_LAYERS} layers to use fused attention"

    print("  [PASS] Full prefill + decode cycle")


# ── Test 6: Multi-request cache clearing ─────────────────────────────

def test_multi_request():
    print_header("Test 6: Multi-Request Cache Clearing")
    reset_global_state()

    D = 128
    KH = 2
    NH = 8
    NUM_LAYERS = 3

    attn_layers = [FakeFlashAttentionImpl(num_kv_heads=KH, head_dim=D)
                   for _ in range(NUM_LAYERS)]
    for a in attn_layers:
        _get_layer_name(a)

    # Request 1: prefill 16 tokens
    for attn in attn_layers:
        layer_name = _get_layer_name(attn)
        key = torch.randn(16, KH, D, device=DEVICE)
        value = torch.randn(16, KH, D, device=DEVICE)
        slot_mapping = torch.arange(16, device=DEVICE)
        _patched_kv_hook(attn, None, key, value, None, slot_mapping)

        meta = FakeAttnMetadata(num_actual_tokens=16, max_query_len=16)
        _patched_forward_hook(attn, None, key, key, value, None, meta, None, None, None)
        _patched_forward_post_hook(attn, layer_name)

    req1_tokens = {_get_layer_name(a): _shadow_cache[_get_layer_name(a)].seq_len
                   for a in attn_layers}
    print(f"  Request 1 prefill: {req1_tokens}")

    # Simulate at least one decode step (this sets _last_was_decode = True)
    decode_meta = FakeAttnMetadata(num_actual_tokens=1, max_query_len=1)
    for attn in attn_layers:
        decode_q = torch.randn(1, NH, D, device=DEVICE)
        _patched_forward_hook(attn, None, decode_q, None, None, None,
                               decode_meta, None, None, None)
    print(f"  Request 1 decode: done")

    # Request 2: prefill should clear cache (new request after decode)
    for attn in attn_layers:
        layer_name = _get_layer_name(attn)
        key = torch.randn(8, KH, D, device=DEVICE)
        value = torch.randn(8, KH, D, device=DEVICE)
        slot_mapping = torch.arange(8, device=DEVICE)
        _patched_kv_hook(attn, None, key, value, None, slot_mapping)

        meta = FakeAttnMetadata(num_actual_tokens=8, max_query_len=8)
        _patched_forward_hook(attn, None, key, key, value, None, meta, None, None, None)
        _patched_forward_post_hook(attn, layer_name)

    req2_tokens = {_get_layer_name(a): _shadow_cache[_get_layer_name(a)].seq_len
                   for a in attn_layers}
    print(f"  Request 2: {req2_tokens}")

    # Request 2 should have 8 tokens, NOT 16+8=24
    for a in attn_layers:
        name = _get_layer_name(a)
        assert _shadow_cache[name].seq_len == 8, \
            f"{name}: expected 8 tokens after new request, got {_shadow_cache[name].seq_len}"

    print("  [PASS] Cache cleared between requests")


# ── Test 7: Decompression accuracy through shadow cache ──────────────

def test_shadow_cache_accuracy():
    print_header("Test 7: Shadow Cache Compression Accuracy")
    reset_global_state()

    D = 128
    KH = 4
    N = 64

    attn = FakeFlashAttentionImpl(num_kv_heads=KH, head_dim=D)
    _get_layer_name(attn)
    layer_name = _get_layer_name(attn)

    # Create known KV data
    key = torch.randn(N, KH, D, device=DEVICE)
    value = torch.randn(N, KH, D, device=DEVICE)

    # Queue and flush
    slot_mapping = torch.arange(N, device=DEVICE)
    _patched_kv_hook(attn, None, key, value, None, slot_mapping)
    _patched_forward_post_hook(attn, layer_name)

    # Get stacked compressed data
    stacked = _shadow_cache[layer_name].get_stacked()
    assert stacked is not None
    k_radii, k_packed, v_radii, v_packed = stacked

    # Decode and verify accuracy
    k_flat = key.reshape(-1, D).float().contiguous()
    k_decoded = _C.tq_decode(
        k_radii.reshape(-1),
        k_packed.reshape(-1, k_packed.size(-1)),
        D, SEED, BITS
    )

    cos = torch.nn.functional.cosine_similarity(
        k_flat.reshape(1, -1), k_decoded.reshape(1, -1)
    ).item()

    print(f"  {N} tokens × {KH} heads, D={D}")
    print(f"  Compressed shapes: radii={list(k_radii.shape)} packed={list(k_packed.shape)}")
    print(f"  Decode cosine similarity: {cos:.4f}")

    assert cos > 0.95, f"Accuracy too low: {cos}"
    print("  [PASS] Shadow cache accuracy verified")


# ── Test 8: Concurrent layer handling (no cross-contamination) ───────

def test_no_cross_contamination():
    print_header("Test 8: No Cross-Layer Contamination")
    reset_global_state()

    D = 128
    KH = 2
    NUM_LAYERS = 4

    attn_layers = [FakeFlashAttentionImpl(num_kv_heads=KH, head_dim=D)
                   for _ in range(NUM_LAYERS)]
    for a in attn_layers:
        _get_layer_name(a)

    # Give each layer DIFFERENT data with different magnitudes
    for i, attn in enumerate(attn_layers):
        layer_name = _get_layer_name(attn)
        scale = (i + 1) * 10.0
        key = torch.randn(8, KH, D, device=DEVICE) * scale
        value = torch.randn(8, KH, D, device=DEVICE) * scale
        slot_mapping = torch.arange(8, device=DEVICE)

        _patched_kv_hook(attn, None, key, value, None, slot_mapping)
        _patched_forward_post_hook(attn, layer_name)

    # Verify each layer has its own distinct data by checking radius magnitudes
    for i, attn in enumerate(attn_layers):
        name = _get_layer_name(attn)
        stacked = _shadow_cache[name].get_stacked()
        k_radii = stacked[0]
        avg_radius = k_radii.mean().item()
        expected_scale = (i + 1) * 10.0
        # Radii should scale roughly with input magnitude
        print(f"  {name}: avg_radius={avg_radius:.1f} (input scale={expected_scale:.0f})")

    # Check that layer_0 and layer_3 have very different radii
    r0 = _shadow_cache[_get_layer_name(attn_layers[0])].get_stacked()[0].mean().item()
    r3 = _shadow_cache[_get_layer_name(attn_layers[3])].get_stacked()[0].mean().item()
    assert r3 > r0 * 2, f"Layer 3 radius ({r3:.1f}) should be much larger than layer 0 ({r0:.1f})"

    print("  [PASS] No cross-contamination between layers")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    print(f"Device: {torch.cuda.get_device_name(0)}")

    test_layer_tracking()
    test_hybrid_architecture()
    test_kv_hook_deferred()
    test_post_forward_flush()
    test_full_cycle()
    test_multi_request()
    test_shadow_cache_accuracy()
    test_no_cross_contamination()

    print(f"\n{'='*60}")
    print("  ALL INTEGRATION TESTS PASSED")
    print(f"{'='*60}")
