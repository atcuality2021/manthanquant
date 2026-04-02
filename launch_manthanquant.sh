#!/bin/bash
# Launch vLLM with ManthanQuant KV cache compression.
# Usage: bash launch_manthanquant.sh [--model PATH] [--port PORT]
#
# This script sets environment variables that activate the ManthanQuant
# monkey-patch via sitecustomize.py (deployed by MCMS node provisioner).
# No files are modified on disk.

set -e

MODEL_PATH=${1:-~/hf_models/Qwen3.5-35B-A3B}
MODEL_NAME=$(basename "$MODEL_PATH")
PORT=${2:-8200}
VLLM_ENV=~/vllm-env

# Verify manthanquant is available
PYTHONPATH=~/manthanquant:${PYTHONPATH:-}
$VLLM_ENV/bin/python3 -c "import manthanquant; print(f'manthanquant {manthanquant.__version__}')" || {
    echo "ERROR: manthanquant not installed. Run: cd ~/manthanquant && $VLLM_ENV/bin/python3 setup.py build_ext --inplace"
    exit 1
}

export PATH=$VLLM_ENV/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=~/nccl/build/lib:$VLLM_ENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:~/cuda_libs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export PYTHONPATH
export MANTHANQUANT_ENABLED=1

mkdir -p ~/logs
rm -f ~/logs/manthanquant_active.flag /tmp/manthanquant_trace_*.log
LOG=~/logs/vllm-manthanquant-$MODEL_NAME-$PORT.log

echo "Launching vLLM + ManthanQuant on port $PORT..."
nohup $VLLM_ENV/bin/vllm serve "$MODEL_PATH" \
    --port $PORT --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization 0.85 --max-model-len 32768 \
    --trust-remote-code --max-num-seqs 2 --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --enable-prefix-caching \
    > "$LOG" 2>&1 &

echo "PID=$! Log=$LOG"
