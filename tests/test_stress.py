#!/usr/bin/env python3
"""
Comprehensive stress test for ManthanQuant 3-bit KV compression
on Qwen3.5-35B-A3B (llm3, DGX Spark GB10).

Tests:
  1. Sustained sequential load (20 requests, growing context)
  2. Concurrent burst (6 simultaneous requests)
  3. Long context push (single request with ~4K token prompt)
  4. Rapid fire (20 short requests as fast as possible)
  5. Multi-turn deep conversation (10 turns, accumulating context)
  6. Mixed workload (short + long + tool calls interleaved)
  7. Error recovery (malformed + valid requests)
"""

import json, time, sys, os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

VLLM_KEY = "mk-86cd7b93f21554926b037db58e61a3c5b58831b7111230dfbcd2a3e31c4e4f8f"
URL = "http://192.168.29.113:8100/v1/chat/completions"

stats = {
    "total_requests": 0,
    "successful": 0,
    "failed": 0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_time": 0,
    "errors": [],
    "latencies": [],
    "throughputs": [],
}


def call(msgs, max_tokens=500, tools=None, timeout=300):
    payload = {
        "model": "Qwen3.5-35B-A3B",
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, headers={
        "Authorization": f"Bearer {VLLM_KEY}",
        "Content-Type": "application/json",
    })
    start = time.time()
    stats["total_requests"] += 1
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        d = json.loads(resp.read())
        elapsed = time.time() - start

        if "error" in d:
            stats["failed"] += 1
            stats["errors"].append(str(d["error"])[:100])
            return None, elapsed

        u = d.get("usage", {})
        pt = u.get("prompt_tokens", 0)
        ct = u.get("completion_tokens", 0)
        tps = ct / elapsed if elapsed > 0 else 0

        stats["successful"] += 1
        stats["total_prompt_tokens"] += pt
        stats["total_completion_tokens"] += ct
        stats["total_time"] += elapsed
        stats["latencies"].append(elapsed)
        stats["throughputs"].append(tps)

        return {"pt": pt, "ct": ct, "tps": tps, "time": elapsed,
                "content": (d["choices"][0]["message"].get("content") or ""),
                "tool_calls": d["choices"][0]["message"].get("tool_calls", [])}, elapsed
    except Exception as e:
        elapsed = time.time() - start
        stats["failed"] += 1
        stats["errors"].append(str(e)[:100])
        return None, elapsed


def heading(title):
    print(f"\n{'━'*70}")
    print(f"  {title}")
    print(f"{'━'*70}")


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Sustained sequential load
# ═══════════════════════════════════════════════════════════════════
heading("TEST 1: Sustained Sequential Load (20 requests, growing output)")
topics = [
    "GPU memory architecture", "CUDA unified memory", "KV cache in transformers",
    "PagedAttention algorithm", "Lloyd-Max quantization theory", "MoE routing",
    "speculative decoding with MTP", "FlashAttention v2 algorithm",
    "tensor parallelism for inference", "NCCL collective operations",
    "Redis caching for LLM", "WebSocket streaming for chat", "rate limiting",
    "JWT authentication", "Prometheus GPU monitoring", "ONNX model optimization",
    "quantization-aware training", "LoRA fine-tuning", "RLHF pipeline",
    "distributed training with DeepSpeed",
]
for i, topic in enumerate(topics):
    tok = 200 + i * 50  # 200 to 1150 tokens
    r, t = call([{"role": "user", "content": f"Explain {topic} in {tok//4} words."}], max_tokens=tok)
    if r:
        print(f"  [{i+1:2d}/20] {r['pt']:4d}p+{r['ct']:5d}c | {t:5.1f}s | {r['tps']:5.1f}t/s | {topic[:30]}")
    else:
        print(f"  [{i+1:2d}/20] FAILED | {t:.1f}s | {topic[:30]}")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Concurrent burst
# ═══════════════════════════════════════════════════════════════════
heading("TEST 2: Concurrent Burst (6 simultaneous requests)")
prompts = [
    "Write a Python red-black tree implementation.",
    "Explain quantum computing in 300 words.",
    "Design a distributed key-value store architecture.",
    "Compare TCP vs UDP for real-time streaming.",
    "Write a CUDA kernel for matrix multiplication.",
    "Explain how transformers handle long-range dependencies.",
]
start_burst = time.time()
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = {}
    for i, p in enumerate(prompts):
        f = pool.submit(call, [{"role": "user", "content": p}], 800)
        futures[f] = i
    for f in as_completed(futures):
        i = futures[f]
        r, t = f.result()
        if r:
            print(f"  [Req {i+1}] {r['pt']:4d}p+{r['ct']:5d}c | {t:5.1f}s | {r['tps']:5.1f}t/s")
        else:
            print(f"  [Req {i+1}] FAILED | {t:.1f}s")
burst_time = time.time() - start_burst
print(f"  Total burst time: {burst_time:.1f}s (all 6 concurrent)")


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Long context push
# ═══════════════════════════════════════════════════════════════════
heading("TEST 3: Long Context Push (~4K token prompt)")
long_prompt = """You are writing a comprehensive technical specification document.

SECTION 1: GPU MEMORY ARCHITECTURE
Modern GPU memory systems use a hierarchical design with registers, shared memory (L1),
L2 cache, and global DRAM. NVIDIA's GB10 architecture introduces unified memory where
CPU and GPU share the same 121GB physical RAM through a single memory controller. This
eliminates the need for explicit data transfers between CPU and GPU, but introduces new
challenges for memory management.

The memory bandwidth of GB10 is approximately 250 GB/s, shared between CPU and GPU access
paths. This is lower than discrete HBM-based GPUs (A100: 2TB/s, H100: 3.35TB/s) but the
unified architecture provides unique advantages for LLM inference where model weights can
be accessed without copy overhead.

SECTION 2: KV CACHE MANAGEMENT
In autoregressive transformer inference, the KV (Key-Value) cache stores intermediate
attention computation results. For each token generated, every attention layer stores a
key vector and a value vector. The memory cost grows as:

  KV_size = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × dtype_size

For Qwen3.5-35B-A3B (11 attention layers, 2 KV heads, 256 head_dim, bf16):
  Per token: 2 × 11 × 2 × 256 × 2 = 22,528 bytes = 22 KB
  At 32K context: 22 KB × 32,768 = 704 MB per sequence
  With batch_size=6: 4.2 GB of KV cache alone

vLLM uses PagedAttention to manage this memory efficiently. Instead of pre-allocating
contiguous memory for each sequence, it uses a page table that maps virtual KV cache
blocks to physical memory blocks, similar to OS virtual memory.

SECTION 3: COMPRESSION APPROACHES
Several approaches exist for reducing KV cache memory:

1. KIVI (2-4x): Stores KV cache in 2-4 bit integers with per-channel scales
2. Gear (2-4x): Uses low-rank approximation combined with quantization
3. MiniCache (2x): Caches only important tokens, evicting less relevant ones
4. TurboQuant/ManthanQuant (5.12x): Per-vector Lloyd-Max 3-bit quantization with
   L2 radius preservation. Achieves highest compression while maintaining cos_sim > 0.98.

Our ManthanQuant approach stores each 256-dim vector as:
  - radius: float32 (4 bytes) — L2 norm
  - packed: uint8[96] (96 bytes) — 3-bit Lloyd-Max centroid indices bit-packed

This gives 512 bytes → 100 bytes = 5.12x compression per vector.

TASK: Based on the above technical context, write a detailed implementation plan for
deploying KV cache compression in a production 3-node GPU cluster. Cover: architecture,
data flow, error handling, monitoring, rollback strategy, and performance testing plan.
Include Python code for the critical compression/decompression path."""

r, t = call([{"role": "user", "content": long_prompt}], max_tokens=3000, timeout=300)
if r:
    print(f"  {r['pt']:4d}p + {r['ct']:5d}c = {r['pt']+r['ct']:5d}t | {t:5.1f}s | {r['tps']:5.1f}t/s")
    print(f"  Response: {len(r['content'])} chars")
else:
    print(f"  FAILED after {t:.1f}s")


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Rapid fire
# ═══════════════════════════════════════════════════════════════════
heading("TEST 4: Rapid Fire (20 short requests, minimal tokens)")
start_rapid = time.time()
for i in range(20):
    r, t = call([{"role": "user", "content": f"What is {i+1} × {i+7}?"}], max_tokens=20, timeout=30)
    status = f"{r['ct']}t {t:.1f}s" if r else "FAIL"
    sys.stdout.write(f"  [{i+1:2d}] {status}  ")
    if (i + 1) % 5 == 0:
        print()
rapid_time = time.time() - start_rapid
print(f"  Total: {rapid_time:.1f}s for 20 requests ({20/rapid_time:.1f} req/s)")


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Multi-turn deep conversation (10 turns)
# ═══════════════════════════════════════════════════════════════════
heading("TEST 5: Multi-turn Deep Conversation (10 turns, accumulating context)")
conversation = [
    {"role": "system", "content": "You are a GPU systems expert. Build on each answer."},
]
turns = [
    "What is a GPU and how does it differ from a CPU?",
    "Explain the GPU memory hierarchy in detail.",
    "How does CUDA programming model map to this hardware?",
    "What is FlashAttention and why is it important for transformers?",
    "Explain KV cache and why it's the bottleneck in LLM inference.",
    "How does vLLM PagedAttention solve the KV cache fragmentation problem?",
    "What is speculative decoding and how does MTP work?",
    "Explain 3-bit Lloyd-Max quantization for KV compression mathematically.",
    "How does unified memory on GB10 change the compression strategy?",
    "Design a complete production system combining all these optimizations.",
]

for i, turn in enumerate(turns):
    conversation.append({"role": "user", "content": turn})
    r, t = call(conversation, max_tokens=600, timeout=120)
    if r:
        conversation.append({"role": "assistant", "content": r["content"][:300]})
        print(f"  Turn {i+1:2d}: {r['pt']:5d}p+{r['ct']:4d}c | {t:5.1f}s | {r['tps']:5.1f}t/s | ctx={r['pt']+r['ct']}t")
    else:
        conversation.append({"role": "assistant", "content": "Error."})
        print(f"  Turn {i+1:2d}: FAILED | {t:.1f}s")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: Mixed workload
# ═══════════════════════════════════════════════════════════════════
heading("TEST 6: Mixed Workload (short + long + tools)")
TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "Search web",
     "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}},
    {"type": "function", "function": {"name": "calc", "description": "Calculate",
     "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
]
mixed = [
    ({"max_tokens": 20}, [{"role": "user", "content": "Hi, how are you?"}]),
    ({"max_tokens": 1000}, [{"role": "user", "content": "Write a detailed essay about GPU evolution from GeForce 256 to GB10."}]),
    ({"max_tokens": 200, "tools": TOOLS}, [{"role": "user", "content": "Search for NVIDIA GB10 specs and calculate 121GB / 5.12."}]),
    ({"max_tokens": 50}, [{"role": "user", "content": "Translate 'Hello World' to Japanese, Hindi, and Arabic."}]),
    ({"max_tokens": 1500}, [{"role": "user", "content": "Write a complete Python async HTTP server with routing, middleware, and error handling."}]),
    ({"max_tokens": 100, "tools": TOOLS}, [{"role": "user", "content": "Calculate (381000 * 5.12) - 381000 and search for vLLM PagedAttention paper."}]),
    ({"max_tokens": 30}, [{"role": "user", "content": "What day is it today?"}]),
    ({"max_tokens": 2000}, [{"role": "user", "content": "Compare NVIDIA A100, H100, and GB10 for LLM inference. Include memory bandwidth, FLOPS, pricing, and power consumption. Show calculations."}]),
]
for i, (opts, msgs) in enumerate(mixed):
    tools = opts.pop("tools", None)
    r, t = call(msgs, tools=tools, **opts)
    if r:
        tc = len(r.get("tool_calls", []))
        tc_info = f" +{tc}tools" if tc else ""
        print(f"  [{i+1}] {r['pt']:4d}p+{r['ct']:5d}c | {t:5.1f}s | {r['tps']:5.1f}t/s{tc_info}")
    else:
        print(f"  [{i+1}] FAILED | {t:.1f}s")


# ═══════════════════════════════════════════════════════════════════
# TEST 7: Error recovery
# ═══════════════════════════════════════════════════════════════════
heading("TEST 7: Error Recovery (bad request → good request)")
# Bad request (empty messages)
r, t = call([], max_tokens=10, timeout=10)
print(f"  Bad request (empty): {'FAILED as expected' if not r else 'UNEXPECTED OK'} | {t:.1f}s")

# Good request after bad
r, t = call([{"role": "user", "content": "Are you still working after that bad request?"}], max_tokens=50)
print(f"  Recovery request:    {'OK' if r else 'FAILED'} | {t:.1f}s | {r['ct'] if r else 0}t")


# ═══════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════
import numpy as np
latencies = np.array(stats["latencies"]) if stats["latencies"] else np.array([0])
throughputs = np.array(stats["throughputs"]) if stats["throughputs"] else np.array([0])

print(f"\n{'='*70}")
print(f"  STRESS TEST FINAL REPORT")
print(f"{'='*70}")
print(f"  Total requests:     {stats['total_requests']}")
print(f"  Successful:         {stats['successful']}")
print(f"  Failed:             {stats['failed']}")
print(f"  Success rate:       {stats['successful']/max(stats['total_requests'],1)*100:.1f}%")
print(f"")
print(f"  Total tokens:       {stats['total_prompt_tokens']+stats['total_completion_tokens']:,}")
print(f"    Prompt tokens:    {stats['total_prompt_tokens']:,}")
print(f"    Completion tokens:{stats['total_completion_tokens']:,}")
print(f"  Total time:         {stats['total_time']:.1f}s")
print(f"")
print(f"  Throughput:")
print(f"    Mean:             {np.mean(throughputs):.1f} tok/s")
print(f"    Median:           {np.median(throughputs):.1f} tok/s")
print(f"    P95:              {np.percentile(throughputs, 95):.1f} tok/s")
print(f"    Min:              {np.min(throughputs):.1f} tok/s")
print(f"    Max:              {np.max(throughputs):.1f} tok/s")
print(f"")
print(f"  Latency:")
print(f"    Mean:             {np.mean(latencies):.1f}s")
print(f"    Median:           {np.median(latencies):.1f}s")
print(f"    P95:              {np.percentile(latencies, 95):.1f}s")
print(f"    P99:              {np.percentile(latencies, 99):.1f}s")
print(f"    Max:              {np.max(latencies):.1f}s")
print(f"")
if stats["errors"]:
    print(f"  Errors ({len(stats['errors'])}):")
    for e in stats["errors"][:5]:
        print(f"    - {e}")
print(f"{'='*70}")
