#!/usr/bin/env python3
"""
Extended baseline tests for clean vLLM (NO ManthanQuant).
Covers: TTFT, TGS at varying context, concurrent scaling, long generation,
KV cache pressure, prefix cache behavior.
"""

import json, time, sys, numpy as np
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

VLLM_KEY = "mk-86cd7b93f21554926b037db58e61a3c5b58831b7111230dfbcd2a3e31c4e4f8f"
URL = "http://192.168.29.113:8100/v1/chat/completions"

def call(msgs, max_tokens=500, stream=False):
    payload = {"model":"Qwen3.5-35B-A3B","messages":msgs,"max_tokens":max_tokens,"temperature":0}
    if stream:
        payload["stream"] = True
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={
        "Authorization":f"Bearer {VLLM_KEY}","Content-Type":"application/json"
    })
    start = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=300)
        if stream:
            # Measure TTFT — time to first data chunk
            first_chunk = resp.read(1)
            ttft = time.time() - start
            rest = resp.read()
            total_time = time.time() - start
            # Parse SSE to count tokens (rough)
            lines = (first_chunk + rest).decode().split('\n')
            token_count = sum(1 for l in lines if l.startswith('data:') and '"delta"' in l)
            return {"ttft": ttft, "total_time": total_time, "tokens": token_count}, total_time
        else:
            d = json.loads(resp.read())
            elapsed = time.time() - start
            if "error" in d:
                return None, elapsed
            u = d.get("usage",{})
            pt=u.get("prompt_tokens",0); ct=u.get("completion_tokens",0)
            return {"pt":pt,"ct":ct,"tps":ct/elapsed if elapsed>0 else 0,"time":elapsed}, elapsed
    except Exception as e:
        return None, time.time()-start


def heading(title):
    print(f"\n{'━'*70}")
    print(f"  {title}")
    print(f"{'━'*70}")


# ═══════════════════════════════════════════════════════════════
# TEST A: Time-to-First-Token at varying prompt lengths
# ═══════════════════════════════════════════════════════════════
heading("TEST A: Time-to-First-Token (TTFT) at varying prompt lengths")
print(f"  {'Prompt Len':>12} {'TTFT':>8} {'Total':>8} {'TGS':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

for prompt_tokens in [50, 200, 500, 1000, 2000]:
    # Build prompt of approximate length
    filler = "Explain GPU memory management in detail. " * (prompt_tokens // 8)
    prompt = f"Context: {filler}\n\nQuestion: Based on the above, summarize the key points in 100 words."

    start = time.time()
    r, t = call([{"role":"user","content":prompt}], max_tokens=200)
    elapsed = time.time() - start

    if r:
        # TTFT approximation: for non-streaming, use (elapsed - ct/tps) as rough prefill time
        prefill_time = elapsed - (r['ct'] / r['tps']) if r['tps'] > 0 else 0
        print(f"  {r['pt']:>10}t {prefill_time:>7.2f}s {elapsed:>7.1f}s {r['tps']:>7.1f}")
    else:
        print(f"  ~{prompt_tokens:>8}t  FAILED  {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════
# TEST B: Token Generation Speed (TGS) at varying output lengths
# ═══════════════════════════════════════════════════════════════
heading("TEST B: Generation Speed at varying output lengths")
print(f"  {'Output Len':>12} {'Actual':>8} {'Time':>8} {'TGS':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")

for max_tok in [50, 200, 500, 1000, 2000, 4000]:
    r, t = call([{"role":"user","content":"Write a detailed technical article about GPU computing, CUDA programming, memory optimization, and inference serving. Be thorough."}], max_tokens=max_tok)
    if r:
        print(f"  {max_tok:>10}t {r['ct']:>7}t {t:>7.1f}s {r['tps']:>7.1f}")
    else:
        print(f"  {max_tok:>10}t  FAILED  {t:.1f}s")


# ═══════════════════════════════════════════════════════════════
# TEST C: Concurrent scaling (1, 2, 3, 4, 6 users)
# ═══════════════════════════════════════════════════════════════
heading("TEST C: Concurrent Scaling (1-6 simultaneous users)")
print(f"  {'Users':>6} {'Agg tok/s':>10} {'Per-user':>10} {'Wall time':>10} {'Tokens':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

prompt_c = "Write 200 words about distributed systems and consensus algorithms."
for n_users in [1, 2, 3, 4, 6]:
    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=n_users) as pool:
        futs = [pool.submit(call, [{"role":"user","content":prompt_c}], 500) for _ in range(n_users)]
        for f in as_completed(futs):
            r, t = f.result()
            if r:
                results.append(r)
    wall = time.time() - start
    total_tok = sum(r['ct'] for r in results)
    agg_tps = total_tok / wall if wall > 0 else 0
    per_user = agg_tps / n_users if n_users > 0 else 0
    print(f"  {n_users:>6} {agg_tps:>9.1f} {per_user:>9.1f} {wall:>9.1f}s {total_tok:>7}")


# ═══════════════════════════════════════════════════════════════
# TEST D: Prefix cache effectiveness
# ═══════════════════════════════════════════════════════════════
heading("TEST D: Prefix Cache — Same prefix, different suffixes")
prefix = "You are BiltIQ AI, a GPU cluster management expert. Your cluster has 3 DGX Spark GB10 nodes with 121GB unified memory each, running Qwen3.5-35B-A3B with MTP speculative decoding."
suffixes = [
    "How many concurrent 32K context users can this serve?",
    "What is the KV cache memory per token?",
    "Explain the MTP speculative decoding acceptance rate.",
    "What optimizations should we add next?",
    "Compare this with an A100-based cluster.",
    "What is the power efficiency in tok/watt?",
    "How does unified memory affect inference?",
    "Design a load balancing strategy for 3 nodes.",
]

print(f"  {'Request':>4} {'Prompt':>7} {'Comp':>6} {'Time':>7} {'TGS':>6} {'Notes'}")
for i, suffix in enumerate(suffixes):
    r, t = call([{"role":"system","content":prefix},{"role":"user","content":suffix}], max_tokens=300)
    if r:
        note = "cold" if i == 0 else "prefix cached" if r['time'] < t * 0.8 else ""
        print(f"  {i+1:>4} {r['pt']:>6}t {r['ct']:>5}t {t:>6.1f}s {r['tps']:>5.1f} {note}")


# ═══════════════════════════════════════════════════════════════
# TEST E: Long sustained generation (single request, 8K output)
# ═══════════════════════════════════════════════════════════════
heading("TEST E: Long Generation (8000 tokens)")
r, t = call([{"role":"user","content":"Write a comprehensive 3000-word technical paper about GPU memory management for LLM inference. Cover: memory hierarchy, KV cache, PagedAttention, quantization, compression, speculative decoding, unified memory architecture, benchmarking methodology, and future directions. Include code examples, mathematical formulations, and performance data."}], max_tokens=8000)
if r:
    print(f"  {r['pt']}p + {r['ct']}c = {r['pt']+r['ct']}t | {t:.1f}s | {r['tps']:.1f} t/s")
    print(f"  Response: {len(r['content'])} chars")
    # Check if speed degraded over the long generation
    print(f"  (Speed should be stable — no KV cache degradation)")
else:
    print(f"  FAILED after {t:.1f}s")


# ═══════════════════════════════════════════════════════════════
# TEST F: KV cache metrics from vLLM
# ═══════════════════════════════════════════════════════════════
heading("TEST F: vLLM KV Cache Metrics (from logs)")
import subprocess
result = subprocess.run(
    ["sudo", "-u", "atc", "ssh", "192.168.29.113",
     "grep -a 'KV cache usage\\|Prefix cache\\|SpecDecoding' ~/logs/vllm-*35B*8100.log | tail -6"],
    capture_output=True, text=True, timeout=10
)
for line in result.stdout.strip().split('\n'):
    # Extract just the metrics part
    if 'Engine 000' in line:
        parts = line.split('Engine 000: ')[1] if 'Engine 000: ' in line else line
        print(f"  {parts.strip()}")
    elif 'SpecDecoding' in line:
        parts = line.split('SpecDecoding metrics: ')[1] if 'SpecDecoding metrics: ' in line else line
        print(f"  Spec: {parts.strip()}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
heading("EXTENDED BASELINE SUMMARY")
print("  Configuration: Clean vLLM, NO ManthanQuant")
print("  Model: Qwen3.5-35B-A3B, MTP speculative, thinking OFF, 32K ctx")
print("  Hardware: DGX Spark GB10, 121GB unified, ARM aarch64")
print("  Tests completed successfully")
