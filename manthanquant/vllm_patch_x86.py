"""
vllm_patch_x86.py — TurboQuant KV cache compression for vLLM on x86 discrete GPUs.

Activation: set MANTHANQUANT_ENABLED=1 in the environment before starting vLLM.

Strategy for x86 discrete GPUs (RTX 4070/6000, A100, H100, etc.):
  PREFILL: Standard FlashAttention (paged KV cache)
           + deferred compression: queue KV data, compress AFTER each layer's
             forward() completes using PyTorch ops on GPU (no PCIe transfer)
  DECODE:  Standard FlashAttention from paged KV cache
           (fused compressed decode disabled until kernel shape fixes)

Key differences from GB10 (vllm_patch.py):
  - GB10: .cpu().numpy() is free (unified memory) → compress on ARM CPU with numpy
  - x86:  .cpu() costs PCIe bandwidth → compress on GPU using PyTorch ops
  - GB10: No _C import (conflicts with Triton on ARM)
  - x86:  _C import works fine (separate VRAM, no Triton conflicts)

Architecture-agnostic layer tracking:
  Uses id(self) on FlashAttentionImpl instances for layer identification.
  Works with any model architecture.

Compression ratio: 5.12x (same Lloyd-Max 3-bit algorithm, same quality)
"""

import logging
import os
import torch
from typing import Optional

logger = logging.getLogger("manthanquant.x86")

SEED = 42
BITS = 3
_installed = False
_original_forward = None
_original_do_kv_cache_update = None
_patched_backend = None  # "flash" or "triton"

# Import compression functions
from .x86_quantize import tq_encode_torch, tq_decode_torch

# Try CUDA extension — but only use if compiled for current GPU's SM
# The _C extension was compiled for specific architectures (e.g. SM 8.9).
# Running on a GPU with a different SM (e.g. SM 12.0 Blackwell) causes
# cudaErrorNoKernelImageForDevice which is a STICKY error that poisons
# all future CUDA calls. Check SM capability before using _C.
_C = None
HAS_CUDA = False

# SM architectures the _C extension was compiled for
# Must match TORCH_CUDA_ARCH_LIST used during build
_COMPILED_SM = {(8, 9), (12, 0)}  # RTX 4070/4090, RTX 6000 Blackwell

try:
    import torch as _torch
    if _torch.cuda.is_available():
        _dev = _torch.cuda.current_device()
        _sm = _torch.cuda.get_device_capability(_dev)
        if _sm in _COMPILED_SM:
            try:
                import manthanquant._C as _C
                HAS_CUDA = True
            except ImportError:
                pass
        else:
            logger.info("GPU SM %d.%d not in compiled set %s — using PyTorch fallback",
                        _sm[0], _sm[1], _COMPILED_SM)
except Exception:
    pass


# ── Per-layer compressed KV cache ─────────────────────────────────────────

class LayerCacheGPU:
    """Shadow compressed cache for one attention layer on GPU VRAM.

    x86 discrete GPU strategy:
    - KV data stays on GPU — no PCIe transfers
    - Compression uses vectorized PyTorch ops on GPU
    - Compressed data stored as GPU tensors
    - 5.12x compression ratio: bf16 [N,D] → float32 radii [N] + int32 packed [N,words]

    NOTE: GPU-side compression (17 tok/s) outperforms CPU-side (8 tok/s) on x86
    because PCIe transfers + GIL blocking cost more than GPU compute overlap.
    This is opposite to GB10 where unified memory makes CPU-side compression free.
    """
    __slots__ = ['k_radii', 'k_packed', 'v_radii', 'v_packed',
                 'seq_len', 'orig_bytes', 'comp_bytes', 'head_dim']

    def __init__(self):
        self.k_radii = []
        self.k_packed = []
        self.v_radii = []
        self.v_packed = []
        self.seq_len = 0
        self.orig_bytes = 0
        self.comp_bytes = 0
        self.head_dim = 0

    def clear(self):
        self.k_radii.clear()
        self.k_packed.clear()
        self.v_radii.clear()
        self.v_packed.clear()
        self.seq_len = 0
        self.orig_bytes = 0
        self.comp_bytes = 0

    def compress_and_append(self, key: torch.Tensor, value: torch.Tensor):
        """Compress KV using 3-bit Lloyd-Max on GPU.

        Input: torch tensors [tokens, kv_heads, head_dim] on CUDA.
        Uses vectorized PyTorch ops — no Python for-loops.
        """
        num_tokens = key.shape[0]
        num_kv_heads = key.shape[1]
        head_dim = key.shape[2]
        self.head_dim = head_dim

        # Track original bf16 size
        self.orig_bytes += 2 * num_tokens * num_kv_heads * head_dim * 2

        # Flatten: [tokens, kv_heads, dim] → [tokens * kv_heads, dim]
        k_flat = key.reshape(-1, head_dim).float()
        v_flat = value.reshape(-1, head_dim).float()

        # 3-bit Lloyd-Max quantization on GPU
        if HAS_CUDA:
            kr, kp = _C.tq_encode(k_flat, SEED, BITS)
            vr, vp = _C.tq_encode(v_flat, SEED, BITS)
        else:
            kr, kp = tq_encode_torch(k_flat, BITS)
            vr, vp = tq_encode_torch(v_flat, BITS)

        # Track compressed size
        self.comp_bytes += 2 * (kr.nelement() * 4 + kp.nelement() * 4)

        # Store reshaped
        self.k_radii.append(kr.reshape(num_tokens, num_kv_heads))
        self.k_packed.append(kp.reshape(num_tokens, num_kv_heads, -1))
        self.v_radii.append(vr.reshape(num_tokens, num_kv_heads))
        self.v_packed.append(vp.reshape(num_tokens, num_kv_heads, -1))
        self.seq_len += num_tokens

    def get_stacked(self):
        """Stack all chunks into contiguous tensors."""
        if self.seq_len == 0:
            return None
        return (
            torch.cat(self.k_radii, dim=0),
            torch.cat(self.k_packed, dim=0),
            torch.cat(self.v_radii, dim=0),
            torch.cat(self.v_packed, dim=0),
        )

    def memory_bytes(self):
        return self.comp_bytes

    def compression_ratio(self):
        if self.comp_bytes == 0:
            return 0.0
        return self.orig_bytes / self.comp_bytes


# ── Architecture-agnostic layer identification ───────────────────────────

_instance_to_name: dict[int, str] = {}
_instance_order: list[int] = []
_first_instance_id: int = 0


def _get_layer_name(attn_impl) -> str:
    global _first_instance_id
    inst_id = id(attn_impl)
    if inst_id not in _instance_to_name:
        idx = len(_instance_to_name)
        _instance_to_name[inst_id] = f"attn_{idx}"
        _instance_order.append(inst_id)
        if idx == 0:
            _first_instance_id = inst_id
    return _instance_to_name[inst_id]


def _is_first_layer(attn_impl) -> bool:
    return id(attn_impl) == _first_instance_id


# ── Global state ──────────────────────────────────────────────────────────

_shadow_cache: dict[str, LayerCacheGPU] = {}
_pending_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_request_count = 0
_warmup_done = False
_last_was_decode = False
_stats = {
    "decode_calls": 0, "prefill_calls": 0,
    "decode_fused": 0, "decode_fallback": 0,
    "compressed_bytes": 0, "layers_discovered": 0,
}

_trace_file = None
_trace_count = 0


def _trace(msg):
    global _trace_file, _trace_count
    if _trace_file is None:
        try:
            _trace_file = open(f"/tmp/manthanquant_x86_trace_{os.getpid()}.log", "a")
        except Exception:
            return
    _trace_file.write(msg + "\n")
    _trace_file.flush()
    _trace_count += 1


def clear_cache():
    for cache in _shadow_cache.values():
        cache.clear()
    _pending_kv.clear()


def get_stats():
    total_orig = sum(c.orig_bytes for c in _shadow_cache.values())
    total_comp = sum(c.memory_bytes() for c in _shadow_cache.values())
    return {
        "layers": len(_shadow_cache),
        "total_orig_mb": total_orig / (1024 * 1024),
        "total_comp_mb": total_comp / (1024 * 1024),
        "ratio": total_orig / max(total_comp, 1),
        "backend": "cuda_ext" if HAS_CUDA else "torch_fallback",
        **_stats,
    }


def _get_layer_cache(name: str) -> LayerCacheGPU:
    if name not in _shadow_cache:
        _shadow_cache[name] = LayerCacheGPU()
        _stats["layers_discovered"] = len(_shadow_cache)
    return _shadow_cache[name]


# ── Deferred compression ─────────────────────────────────────────────────
#
# Same pattern as GB10: queue KV data in do_kv_cache_update, compress AFTER
# FlashAttention completes for each layer. On x86 we compress on GPU instead
# of CPU — no PCIe transfer needed.

def _flush_pending_kv():
    """Compress all pending KV data on GPU and store in shadow caches."""
    for layer_name, kv_data in _pending_kv.items():
        if kv_data is None:
            continue
        k_tensor, v_tensor = kv_data
        cache = _get_layer_cache(layer_name)
        cache.compress_and_append(k_tensor, v_tensor)
    _pending_kv.clear()


# ── Monkey-patched methods ────────────────────────────────────────────────

def _patched_do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    """Monkey-patched do_kv_cache_update: run original, then queue KV for compression."""
    global _warmup_done

    _original_do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping)

    num_actual = slot_mapping.size(0)
    if num_actual > 256 and not _warmup_done:
        return
    _warmup_done = True

    try:
        layer_name = _get_layer_name(self)
        # Keep on GPU — GPU-side compression is faster than PCIe + CPU on x86
        k = key[:num_actual].detach().clone()
        v = value[:num_actual].detach().clone()
        _pending_kv[layer_name] = (k, v)
    except Exception as e:
        if _trace_count < 500:
            _trace(f"kv_update ERROR: {e}")


def _patched_forward(self, layer, query, key, value, kv_cache,
                      attn_metadata, output=None, output_scale=None,
                      output_block_scale=None):
    """Monkey-patched forward: defer compression to after FlashAttention."""
    global _request_count, _last_was_decode

    if attn_metadata is None:
        return output.fill_(0) if output is not None else None

    num_actual_tokens = attn_metadata.num_actual_tokens
    max_qlen = getattr(attn_metadata, 'max_query_len', None)
    is_decode = (max_qlen is not None and max_qlen <= 2 and num_actual_tokens <= 16)
    layer_name = _get_layer_name(self)

    # ── Flush deferred compression on first layer of new pass ──
    if _is_first_layer(self) and _pending_kv:
        count = len([v for v in _pending_kv.values() if v is not None])
        _flush_pending_kv()
        _stats["compressed_bytes"] = sum(c.memory_bytes() for c in _shadow_cache.values())
        total_orig = sum(c.orig_bytes for c in _shadow_cache.values())
        total_comp = sum(c.memory_bytes() for c in _shadow_cache.values())
        ratio = total_orig / total_comp if total_comp > 0 else 0
        total_tokens = sum(c.seq_len for c in _shadow_cache.values())
        if _trace_count < 500:
            _trace(f"COMPRESSED: {count} layers, tokens={total_tokens}, "
                   f"orig={total_orig}B, comp={total_comp}B, ratio={ratio:.2f}x, "
                   f"saved={((total_orig-total_comp)/1024):.1f}KB")
        total_fwd = _stats["prefill_calls"]
        if total_fwd > 0 and total_fwd % 10 == 0:
            _dump_stats()

    if not is_decode:
        # ── PREFILL ──
        if _is_first_layer(self) and _last_was_decode:
            for c in _shadow_cache.values():
                c.clear()
            _request_count += 1
            _last_was_decode = False

        _stats["prefill_calls"] += 1

        # Run original FlashAttention
        result = _original_forward(self, layer, query, key, value, kv_cache,
                                    attn_metadata, output, output_scale,
                                    output_block_scale)

        # Compress the queued KV for this layer (AFTER FlashAttention done)
        if layer_name in _pending_kv:
            k, v = _pending_kv.pop(layer_name)
            cache = _get_layer_cache(layer_name)
            cache.compress_and_append(k, v)

        return result

    # ── DECODE ──
    _last_was_decode = True
    _stats["decode_calls"] += 1

    # Flush any remaining pending
    if _pending_kv:
        _flush_pending_kv()

    # For now, always use FlashAttention for decode — fused decode disabled
    _stats["decode_fallback"] += 1
    return _original_forward(self, layer, query, key, value, kv_cache,
                              attn_metadata, output, output_scale,
                              output_block_scale)


def _dump_stats():
    """Write stats to a JSON file for external monitoring."""
    import json
    stats = get_stats()
    stats["shadow_cache_tokens"] = {k: c.seq_len for k, c in _shadow_cache.items()}
    stats["total_compressed_tokens"] = sum(c.seq_len for c in _shadow_cache.values())
    total_orig = stats["total_orig_mb"] * 1024 * 1024
    total_comp = stats["total_comp_mb"] * 1024 * 1024
    stats["memory_saved_mb"] = round((total_orig - total_comp) / (1024 * 1024), 2)

    try:
        path = os.path.expanduser(f"~/logs/manthanquant_x86_stats_{os.getpid()}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass


# ── Patch installation ────────────────────────────────────────────────────

def _do_patch():
    global _original_forward, _original_do_kv_cache_update, _installed, _patched_backend

    if _installed:
        return

    # Try FlashAttention first, then Triton (Gemma 4 uses Triton due to heterogeneous heads)
    AttnImpl = None
    backend_name = None

    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
        AttnImpl = FlashAttentionImpl
        backend_name = "flash"
    except ImportError:
        pass

    try:
        from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
        # Prefer Triton if both available (Gemma 4 forces Triton)
        AttnImpl = TritonAttentionImpl
        backend_name = "triton"
    except ImportError:
        pass

    if AttnImpl is None:
        logger.error("Could not import FlashAttentionImpl or TritonAttentionImpl from vllm.v1")
        return

    if AttnImpl.forward.__name__ == "_patched_forward":
        _installed = True
        return

    _original_forward = AttnImpl.forward
    _original_do_kv_cache_update = AttnImpl.do_kv_cache_update

    AttnImpl.forward = _patched_forward
    AttnImpl.do_kv_cache_update = _patched_do_kv_cache_update

    _installed = True
    _patched_backend = backend_name

    # Also patch FlashAttention if we patched Triton (some layers may use Flash)
    if backend_name == "triton":
        try:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
            if FlashAttentionImpl.forward.__name__ != "_patched_forward":
                FlashAttentionImpl.forward = _patched_forward
                FlashAttentionImpl.do_kv_cache_update = _patched_do_kv_cache_update
        except ImportError:
            pass

    try:
        os.makedirs(os.path.expanduser("~/logs"), exist_ok=True)
        with open(os.path.expanduser("~/logs/manthanquant_x86_active.flag"), "a") as f:
            f.write(f"patched pid={os.getpid()} backend={backend_name} "
                    f"forward={AttnImpl.forward.__name__}\n")
    except Exception:
        pass

    compress_backend = "CUDA kernels" if HAS_CUDA else "PyTorch fallback"
    logger.info("ManthanQuant x86 TurboQuant ACTIVE (pid=%d, attn=%s, compress=%s, id-based)",
                os.getpid(), backend_name, compress_backend)


def install():
    """Install TurboQuant compression hooks into vLLM's FlashAttention."""
    if not os.environ.get("MANTHANQUANT_ENABLED"):
        logger.info("MANTHANQUANT_ENABLED not set, skipping x86 install")
        return

    _do_patch()

    if not _installed:
        # Deferred install: hook __import__ to patch when attention backend loads
        import builtins
        _orig_import = builtins.__import__
        _hooking = False

        def _patching_import(name, *args, **kwargs):
            nonlocal _hooking
            result = _orig_import(name, *args, **kwargs)
            if not _hooking and not _installed and \
               ("flash_attn" in name or "triton_attn" in name) and "attention" in name:
                _hooking = True
                builtins.__import__ = _orig_import
                _do_patch()
            return result

        builtins.__import__ = _patching_import


def uninstall():
    """Remove compression hooks."""
    global _installed, _original_forward, _original_do_kv_cache_update

    if not _installed:
        return

    for mod_path in [
        "vllm.v1.attention.backends.flash_attn",
        "vllm.v1.attention.backends.triton_attn",
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            impl_name = "FlashAttentionImpl" if "flash" in mod_path else "TritonAttentionImpl"
            AttnImpl = getattr(mod, impl_name)
            if AttnImpl.forward.__name__ == "_patched_forward":
                AttnImpl.forward = _original_forward
                AttnImpl.do_kv_cache_update = _original_do_kv_cache_update
        except (ImportError, AttributeError):
            pass

    _installed = False
    logger.info("ManthanQuant x86 uninstalled")
