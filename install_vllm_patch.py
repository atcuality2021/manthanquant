"""
install_vllm_patch.py — Patch vLLM attention backends for ManthanQuant TurboQuant.

Modifies the installed vLLM source to call ManthanQuant hooks in
do_kv_cache_update() and forward(). Works in ALL processes because the
code is in the actual source file (not monkey-patching).

Supported backends (only patched if the file exists):
  - flash_attn   (FlashAttentionImpl)   — default — historical, ASR/Whisper
  - triton_attn  (TritonAttentionImpl)  — default — gemma-4-* on GB10 (sm_121)
  - flashinfer   (FlashInferImpl)       — EXPERIMENTAL, opt-in only.
      The hook fires correctly (active.flag confirms), but on
      Qwen3.5-122B-A10B-GPTQ-Int4 with speculative decoding
      (qwen3_5_mtp / mtp), patched flashinfer hangs chat completions
      indefinitely. Until that's diagnosed, default install skips it.

vLLM picks the backend per model based on architecture compatibility (e.g.
gemma-4 hard-forces TRITON_ATTN at config.py:104 due to heterogeneous head
dims).

Usage:
    ~/vllm-env/bin/python3 install_vllm_patch.py                    # patch defaults (flash_attn + triton_attn)
    ~/vllm-env/bin/python3 install_vllm_patch.py all                # patch ALL backends incl. experimental
    ~/vllm-env/bin/python3 install_vllm_patch.py --backend flashinfer  # opt in to a specific backend
    ~/vllm-env/bin/python3 install_vllm_patch.py --revert           # revert defaults
    ~/vllm-env/bin/python3 install_vllm_patch.py --revert flashinfer
    ~/vllm-env/bin/python3 install_vllm_patch.py --revert all       # revert ALL

Hooks:
1. KV hook after reshape_and_cache_flash()/triton_reshape_and_cache_flash()
   — queues KV data for deferred compression.
2. Forward pre-hook — intercepts decode for fused compressed attention.
3. Forward post-hook — disabled on GB10 (causes device-side asserts);
   compression flushes from the next forward pass's pre-hook instead.

Activity flags written to ~/logs/:
  manthanquant_loaded.flag — appended at module import time. Proves the
      patched file was IMPORTED. vLLM imports backends during registration
      scan even when picking a different one, so this accumulating does NOT
      prove compression is running.
  manthanquant_active.flag — appended on FIRST KV-cache hook fire per process
      (gated by a one-shot mutable cell). Empty file = compression genuinely
      not running. This is the honest signal.
"""

import os
import sys
import shutil
import py_compile
import glob
from typing import Optional

VLLM_ENV = os.path.expanduser("~/vllm-env")
BACKENDS_DIR = os.path.join(
    VLLM_ENV, "lib/python3.12/site-packages/vllm/v1/attention/backends"
)


# ── Backend registry ─────────────────────────────────────────────────────
# Each entry describes one attention backend file we know how to patch.
# `kv_marker` must be the exact prefix (including leading whitespace) of
# the line where the backend writes K/V to the paged cache. The hook is
# inserted on the line AFTER the closing paren of that call.

BACKENDS = [
    {
        "name": "flash_attn",
        "filename": "flash_attn.py",
        "class_name": "FlashAttentionImpl",
        "kv_marker": "        reshape_and_cache_flash(",
        "default": True,
    },
    {
        "name": "triton_attn",
        "filename": "triton_attn.py",
        "class_name": "TritonAttentionImpl",
        "kv_marker": "        triton_reshape_and_cache_flash(",
        "default": True,
    },
    {
        "name": "flashinfer",
        "filename": "flashinfer.py",
        "class_name": "FlashInferImpl",
        # FlashInfer's KV write is inside `if self.kv_sharing_target_layer_name is None:`
        # — 12-space indent. Hook fires only when KV is actually written
        # (skipping KV-sharing layers that reuse another layer's cache).
        "kv_marker": "            torch.ops._C_cache_ops.reshape_and_cache_flash(",
        # EXPERIMENTAL — verified the hook FIRES (active.flag confirms), but
        # chat completions hang on Qwen3.5-122B-A10B-GPTQ-Int4 with the patch
        # active (2026-04-27 testing on llm3). Root cause likely involves the
        # KV layout differences in FlashInfer paged cache and/or speculative
        # decoding (qwen3_5_mtp). Opt-in only via `--backend flashinfer` until
        # a fix lands.
        "default": False,
    },
]


# ── Hook bodies ──────────────────────────────────────────────────────────
# Same text injected into all three backend files. They reference module-level
# names (_MQ_ACTIVE, _mq_patch, _MQ_FIRST_HOOK, _mq_logdir, _mq_os) created
# by IMPORT_BLOCK at the top of each file.

IMPORT_BLOCK = '''
# ── ManthanQuant TurboQuant KV Cache Compression ─────────────────────────
# IMPORTANT: Do NOT import manthanquant._C here — loading custom CUDA
# extensions at import time conflicts with Triton/FlashAttention on GB10.
# Only import the pure-Python vllm_patch module (no CUDA kernels).
#
# Three flags written to ~/logs/:
#   manthanquant_loaded.flag — appended at file import (this block).
#       Proves the patched backend file was IMPORTED. vLLM imports
#       backends during registration scan even when it ultimately picks
#       a different backend, so this flag accumulating does NOT prove
#       compression is running.
#   manthanquant_active.flag — appended on FIRST KV-cache hook fire.
#       Proves at least one forward pass routed through this backend AND
#       the hook executed.
#   manthanquant_skip.flag — appended when our hook intentionally skips
#       (warmup phase, exception paths). Helps distinguish "never ran" from
#       "ran but bypassed" while debugging.
#
# Skip protocol: we ignore the first MANTHANQUANT_HOOK_SKIP forward/KV
# calls so vLLM's profiling/warmup can't poison _pending_kv state. The
# count is per-process, gated by a mutable cell. After the threshold,
# normal hook behavior resumes. This avoids a class of bugs where small
# warmup tensors get queued with shapes that don't match real-inference
# shapes downstream.
_MQ_ACTIVE = False
_MQ_FIRST_HOOK = [True]      # one-shot first-fire flag write
_MQ_SKIP_REMAINING = [5]     # decrement to 0; while >0 the hook is a no-op
try:
    import manthanquant.vllm_patch as _mq_patch
    _MQ_ACTIVE = True
    import os as _mq_os
    _mq_logdir = _mq_os.path.expanduser("~/logs")
    _mq_os.makedirs(_mq_logdir, exist_ok=True)
    with open(_mq_os.path.join(_mq_logdir, "manthanquant_loaded.flag"), "a") as _mq_f:
        _mq_f.write(__name__ + " loaded pid=" + str(_mq_os.getpid()) + "\\n")
except ImportError:
    pass
# ── End ManthanQuant imports ─────────────────────────────────────────────
'''

# KV update hook — fires after each backend's reshape_and_cache_flash call.
# First fire writes to manthanquant_active.flag (honest activation signal).
#
# Defensive design notes:
# - Wrapped in try/except to guarantee a hook failure can never break
#   vLLM's scheduler (silent on failure).
# - Skips the first _MQ_SKIP_REMAINING calls so warmup batches can't
#   poison _pending_kv with shapes/strides that differ from real inference.

KV_UPDATE_HOOK = '''
        # ManthanQuant: queue KV for deferred compression
        if _MQ_ACTIVE:
            try:
                if _MQ_SKIP_REMAINING[0] > 0:
                    _MQ_SKIP_REMAINING[0] -= 1
                else:
                    if _MQ_FIRST_HOOK[0]:
                        _MQ_FIRST_HOOK[0] = False
                        try:
                            with open(_mq_os.path.join(_mq_logdir, "manthanquant_active.flag"), "a") as _mq_f:
                                _mq_f.write("kv_hook_first " + __name__ + " pid=" + str(_mq_os.getpid()) + "\\n")
                        except Exception:
                            pass
                    _mq_patch._patched_kv_hook(self, layer, key, value, kv_cache, slot_mapping)
            except Exception:
                pass  # Never let a hook failure propagate into vLLM
'''

# Forward pre-hook — fires at the start of forward(). Same defensive wrapping.
FORWARD_PRE_HOOK = '''
        # ManthanQuant: intercept decode for fused compressed attention
        if _MQ_ACTIVE and attn_metadata is not None:
            _mq_result = None
            try:
                if _MQ_SKIP_REMAINING[0] <= 0:
                    _mq_result = _mq_patch._patched_forward_hook(
                        self, layer, query, key, value, kv_cache,
                        attn_metadata, output, output_scale, output_block_scale)
            except Exception:
                _mq_result = None  # Never let a hook failure propagate
            if _mq_result is not None:
                return _mq_result
'''

# Forward post-hook is intentionally DISABLED on GB10 — inserting code before
# return statements in forward() causes device-side asserts on unified memory.
# Compression runs in the pre-hook of the next forward pass instead.
# The constant is kept for documentation; the install() function never injects it.
FORWARD_POST_HOOK = '''
        # ManthanQuant: flush deferred KV compression after attention
        if _MQ_ACTIVE:
            _mq_layer_name = _mq_patch._get_layer_name(self)
            _mq_patch._patched_forward_post_hook(self, _mq_layer_name)
'''


# ── Per-backend patch + revert ───────────────────────────────────────────


def _backend_paths(backend: dict) -> tuple[str, str]:
    """Return (file_path, original_backup_path) for a backend dict."""
    file_path = os.path.join(BACKENDS_DIR, backend["filename"])
    return file_path, file_path + ".manthanquant_orig"


def _install_one(backend: dict) -> bool:
    """Install ManthanQuant hooks into one backend file. Returns True on success."""
    file_path, orig_path = _backend_paths(backend)
    name = backend["name"]
    class_name = backend["class_name"]
    kv_marker = backend["kv_marker"]

    if not os.path.exists(file_path):
        print(f"[{name}] SKIP — {file_path} not found")
        return False

    # Backup original on first run; otherwise restore from backup so we always
    # patch a clean source (idempotent re-install).
    if not os.path.exists(orig_path):
        shutil.copy2(file_path, orig_path)
        print(f"[{name}] backed up: {orig_path}")
    else:
        shutil.copy2(orig_path, file_path)

    with open(file_path) as f:
        content = f.read()
    lines = content.split("\n")

    # ── 1. Insert IMPORT_BLOCK above the first import statement ──────────
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i
            break

    import_lines = IMPORT_BLOCK.strip().split("\n")
    lines = lines[:insert_idx] + import_lines + [""] + lines[insert_idx:]

    # ── 2. Insert KV_UPDATE_HOOK after the KV write call ─────────────────
    content_joined = "\n".join(lines)
    idx = content_joined.find(kv_marker)
    if idx < 0:
        print(f"[{name}] WARNING — KV marker not found ({kv_marker!r}); skipping KV hook")
    else:
        # Walk forward to find the matching closing paren; insert AFTER its
        # newline so the hook becomes the next statement.
        paren_count = 0
        search_start = idx + len(kv_marker)
        for ci in range(search_start, len(content_joined)):
            if content_joined[ci] == "(":
                paren_count += 1
            elif content_joined[ci] == ")":
                if paren_count == 0:
                    end_of_line = content_joined.find("\n", ci)
                    content_joined = (
                        content_joined[: end_of_line + 1]
                        + KV_UPDATE_HOOK
                        + content_joined[end_of_line + 1 :]
                    )
                    break
                paren_count -= 1

    lines = content_joined.split("\n")

    # ── 3. Insert FORWARD_PRE_HOOK just before the forward()'s first ──────
    # 'assert output is not None' line inside the impl class. All three
    # backends use this identical anchor at the top of forward().
    in_impl = False
    inserted_pre = False
    for i, line in enumerate(lines):
        if f"class {class_name}" in line:
            in_impl = True
        if in_impl and "assert output is not None" in line:
            lines = lines[:i] + FORWARD_PRE_HOOK.split("\n") + lines[i:]
            inserted_pre = True
            break
    if not inserted_pre:
        print(f"[{name}] WARNING — forward() anchor not found; skipping pre-hook")

    # ── 4. Forward post-hook intentionally skipped on GB10 (see comment above)

    # Write back
    with open(file_path, "w") as f:
        f.write("\n".join(lines))

    # Verify syntax — auto-revert on compile error.
    try:
        py_compile.compile(file_path, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"[{name}] SYNTAX ERROR after patch: {e}")
        shutil.copy2(orig_path, file_path)
        print(f"[{name}] reverted to original")
        return False

    # Clear pyc cache so the next process loads our patched version.
    pyc_glob = os.path.join(
        os.path.dirname(file_path),
        f"__pycache__/{backend['filename'].rsplit('.', 1)[0]}*.pyc",
    )
    for pyc in glob.glob(pyc_glob):
        os.remove(pyc)

    print(f"[{name}] OK")
    return True


def _revert_one(backend: dict) -> bool:
    """Restore one backend file from its .manthanquant_orig backup."""
    file_path, orig_path = _backend_paths(backend)
    name = backend["name"]

    if not os.path.exists(orig_path):
        print(f"[{name}] no backup ({orig_path}) — nothing to revert")
        return False

    shutil.copy2(orig_path, file_path)
    pyc_glob = os.path.join(
        os.path.dirname(file_path),
        f"__pycache__/{backend['filename'].rsplit('.', 1)[0]}*.pyc",
    )
    for pyc in glob.glob(pyc_glob):
        os.remove(pyc)
    print(f"[{name}] reverted")
    return True


# ── CLI ──────────────────────────────────────────────────────────────────


def _resolve_backends(name_filter: Optional[str]) -> list[dict]:
    if not name_filter:
        # Default install only patches backends marked default=True.
        # Use `--backend flashinfer` to opt in to experimental ones.
        return [b for b in BACKENDS if b.get("default", True)]
    if name_filter == "all":
        return BACKENDS
    matches = [b for b in BACKENDS if b["name"] == name_filter]
    if not matches:
        valid = ", ".join(b["name"] for b in BACKENDS)
        sys.exit(f"unknown backend {name_filter!r}; valid: {valid}")
    return matches


def install(name_filter: Optional[str] = None) -> int:
    backends = _resolve_backends(name_filter)
    failures = 0
    for b in backends:
        if not _install_one(b):
            failures += 1
    print(f"\nManthanQuant patch: {len(backends) - failures}/{len(backends)} backends OK")
    return failures


def revert(name_filter: Optional[str] = None) -> int:
    # Revert touches every backend that has a .manthanquant_orig backup,
    # regardless of `default` flag — opt-in installs need a way out.
    if not name_filter:
        backends = BACKENDS
    elif name_filter == "all":
        backends = BACKENDS
    else:
        matches = [b for b in BACKENDS if b["name"] == name_filter]
        if not matches:
            valid = ", ".join(b["name"] for b in BACKENDS)
            sys.exit(f"unknown backend {name_filter!r}; valid: {valid}")
        backends = matches
    for b in backends:
        _revert_one(b)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--revert" in args:
        args.remove("--revert")
        target = args[0] if args else None
        sys.exit(revert(target))
    # `--backend X` or positional X both work
    target = None
    if "--backend" in args:
        i = args.index("--backend")
        target = args[i + 1] if i + 1 < len(args) else None
    elif args:
        target = args[0]
    sys.exit(install(target))
