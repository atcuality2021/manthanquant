"""
install_vllm_patch.py — Patch vLLM's FlashAttentionImpl for TurboQuant.

Modifies the installed vLLM source to call ManthanQuant hooks in
do_kv_cache_update() and forward(). Works in ALL processes because
the code is in the actual source file (not monkey-patching).

Usage:
    ~/vllm-env/bin/python3 install_vllm_patch.py          # install
    ~/vllm-env/bin/python3 install_vllm_patch.py --revert  # revert

Hooks:
1. KV hook after reshape_and_cache_flash() — queues KV data for deferred compression
2. Forward pre-hook — intercepts decode for fused compressed attention
3. Forward post-hook — flushes deferred compression after prefill FlashAttention
"""

import os
import sys
import shutil
import py_compile

VLLM_ENV = os.path.expanduser("~/vllm-env")
FA_PATH = os.path.join(VLLM_ENV, "lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py")
FA_ORIG = FA_PATH + ".manthanquant_orig"

# ── Import block (top of file) ───────────────────────────────────────────

IMPORT_BLOCK = '''
# ── ManthanQuant TurboQuant KV Cache Compression ─────────────────────────
# IMPORTANT: Do NOT import manthanquant._C here — loading custom CUDA
# extensions at import time conflicts with Triton/FlashAttention on GB10.
# Only import the pure-Python vllm_patch module (no CUDA kernels).
_MQ_ACTIVE = False
try:
    import manthanquant.vllm_patch as _mq_patch
    _MQ_ACTIVE = True
    import os as _mq_os
    _mq_logdir = _mq_os.path.expanduser("~/logs")
    _mq_os.makedirs(_mq_logdir, exist_ok=True)
    with open(_mq_os.path.join(_mq_logdir, "manthanquant_active.flag"), "a") as _mq_f:
        _mq_f.write("flash_attn loaded pid=" + str(_mq_os.getpid()) + "\\n")
except ImportError:
    pass
# ── End ManthanQuant imports ─────────────────────────────────────────────
'''

# ── KV update hook (after reshape_and_cache_flash) ───────────────────────
# Passes self, layer, key, value, kv_cache, slot_mapping to the hook.
# The hook only QUEUES data — no CUDA kernels run here.

KV_UPDATE_HOOK = '''
        # ManthanQuant: queue KV for deferred compression
        if _MQ_ACTIVE:
            _mq_patch._patched_kv_hook(self, layer, key, value, kv_cache, slot_mapping)
'''

# ── Forward pre-hook (start of forward) ──────────────────────────────────
# Returns result if decode handled by compressed attention, None for prefill.

FORWARD_PRE_HOOK = '''
        # ManthanQuant: intercept decode for fused compressed attention
        if _MQ_ACTIVE and attn_metadata is not None:
            _mq_result = _mq_patch._patched_forward_hook(
                self, layer, query, key, value, kv_cache,
                attn_metadata, output, output_scale, output_block_scale)
            if _mq_result is not None:
                return _mq_result
'''

# ── Forward post-hook (end of forward, before return) ────────────────────
# Flushes deferred KV compression AFTER FlashAttention completes.

FORWARD_POST_HOOK = '''
        # ManthanQuant: flush deferred KV compression after FlashAttention
        if _MQ_ACTIVE:
            _mq_layer_name = _mq_patch._get_layer_name(self)
            _mq_patch._patched_forward_post_hook(self, _mq_layer_name)
'''


def install():
    """Install ManthanQuant hooks into vLLM's flash_attn.py."""
    if not os.path.exists(FA_PATH):
        print(f"ERROR: {FA_PATH} not found")
        sys.exit(1)

    # Backup original
    if not os.path.exists(FA_ORIG):
        shutil.copy2(FA_PATH, FA_ORIG)
        print(f"Backed up: {FA_ORIG}")
    else:
        shutil.copy2(FA_ORIG, FA_PATH)

    with open(FA_PATH) as f:
        content = f.read()
    lines = content.split("\n")

    # ── 1. Insert import block ───────────────────────────────────────────
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_idx = i
            break

    import_lines = IMPORT_BLOCK.strip().split("\n")
    lines = lines[:insert_idx] + import_lines + [""] + lines[insert_idx:]
    print("Inserted import block")

    # ── 2. Insert KV hook after reshape_and_cache_flash() ────────────────
    content_joined = "\n".join(lines)
    marker = "        reshape_and_cache_flash("
    idx = content_joined.find(marker)
    if idx >= 0:
        paren_count = 0
        search_start = idx + len(marker)
        for ci in range(search_start, len(content_joined)):
            if content_joined[ci] == "(":
                paren_count += 1
            elif content_joined[ci] == ")":
                if paren_count == 0:
                    end_of_line = content_joined.find("\n", ci)
                    content_joined = (
                        content_joined[:end_of_line + 1] +
                        KV_UPDATE_HOOK +
                        content_joined[end_of_line + 1:]
                    )
                    break
                paren_count -= 1
        print("Inserted KV update hook")
    else:
        print("WARNING: Could not find reshape_and_cache_flash marker")

    # ── 3. Insert forward pre-hook (at start of forward body) ────────────
    # Find 'assert output is not None' inside FlashAttentionImpl.forward()
    # This is the first executable line after the docstring — reliable anchor.
    lines = content_joined.split("\n")
    in_impl = False
    for i, line in enumerate(lines):
        if "class FlashAttentionImpl" in line:
            in_impl = True
        if in_impl and "assert output is not None" in line:
            # Insert hook BEFORE this assert
            lines = lines[:i] + FORWARD_PRE_HOOK.split("\n") + lines[i:]
            print(f"Inserted forward pre-hook at line {i}")
            break
    else:
        print("WARNING: Could not find forward() body anchor (assert output is not None)")

    # ── 4. Insert forward post-hook before return output in forward() ──────
    # The post-hook flushes deferred KV compression AFTER FlashAttention
    # completes for each layer. This is critical — without it, _pending_kv
    # is queued by the KV hook but never compressed into the shadow cache.
    #
    # Strategy: insert before ALL 'return output' lines inside forward()
    # EXCEPT:
    #   - 'return output.fill_(0)' (profiling)
    #   - returns inside _forward_encoder_attention calls
    #   - the ManthanQuant pre-hook return (_mq_result)
    # Work backwards so line indices don't shift.
    in_forward = False
    forward_end = len(lines)
    forward_start = 0
    return_indices = []
    in_impl = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if "class FlashAttentionImpl" in line:
            in_impl = True
        # forward() inside FlashAttentionImpl — the signature spans multiple lines
        if in_impl and not in_forward and "    def forward(" in line:
            in_forward = True
            forward_start = i
            continue
        # Next method definition at same indent level = end of forward()
        if in_forward and line.startswith("    def ") and "forward" not in line:
            forward_end = i
            break
        if in_forward and "return output" in line:
            # Skip profiling, encoder returns, and ManthanQuant pre-hook
            if "return output.fill_" in line:
                continue
            if "_mq_result" in line:
                continue
            if "return self._forward_encoder" in line:
                continue
            return_indices.append(i)

    # GB10: Post-hook disabled — inserting code before return statements
    # causes CUDA device-side asserts on GB10 unified memory.
    # Compression runs in the pre-hook of the next forward pass instead.
    return_indices = []

    post_hook_template = FORWARD_POST_HOOK.strip().split("\n")
    base_indent = "        "  # 8 spaces — template's indent level
    inserted = 0
    for idx in reversed(return_indices):
        # Detect the return line's indentation
        return_line = lines[idx]
        actual_indent = return_line[:len(return_line) - len(return_line.lstrip())]
        # Re-indent the post-hook to match
        adjusted = []
        for pl in post_hook_template:
            stripped_pl = pl.lstrip()
            if not stripped_pl:
                adjusted.append("")
                continue
            # Replace base indent with actual indent
            old_indent = pl[:len(pl) - len(stripped_pl)]
            extra = old_indent[len(base_indent):] if len(old_indent) >= len(base_indent) else ""
            adjusted.append(actual_indent + extra + stripped_pl)
        lines = lines[:idx] + adjusted + [""] + lines[idx:]
        inserted += 1

    # Post-hook disabled on GB10 — inserting code before return statements
    # in forward() causes device-side asserts. Compression runs in the
    # pre-hook of the next forward pass instead.
    return_indices = []  # Override — don't insert post-hooks
    inserted = 0
    if inserted > 0:
        print(f"Inserted forward post-hook before {inserted} return statement(s)")
    else:
        print("Post-hook skipped (GB10 compatibility)")

    # Write back
    with open(FA_PATH, "w") as f:
        f.write("\n".join(lines))

    # Verify syntax
    try:
        py_compile.compile(FA_PATH, doraise=True)
        print("Syntax OK")
    except py_compile.PyCompileError as e:
        print(f"SYNTAX ERROR: {e}")
        shutil.copy2(FA_ORIG, FA_PATH)
        print("Reverted to original")
        sys.exit(1)

    # Clear pycache
    import glob
    for pyc in glob.glob(os.path.join(os.path.dirname(FA_PATH), "__pycache__/flash_attn*.pyc")):
        os.remove(pyc)
        print(f"Removed: {pyc}")

    print("ManthanQuant TurboQuant patch installed successfully")


def revert():
    """Revert to original flash_attn.py."""
    if os.path.exists(FA_ORIG):
        shutil.copy2(FA_ORIG, FA_PATH)
        import glob
        for pyc in glob.glob(os.path.join(os.path.dirname(FA_PATH), "__pycache__/flash_attn*.pyc")):
            os.remove(pyc)
        print("Reverted to original")
    else:
        print("No backup found")


if __name__ == "__main__":
    if "--revert" in sys.argv:
        revert()
    else:
        install()
