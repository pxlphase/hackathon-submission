#!/usr/bin/env bash
# Launch Qwen/Qwen3.8-27B on NVIDIA Brev (1x RTX PRO 6000 96 GB)
# with MTP speculative decoding and qwen3 reasoning parser.
#
# Usage on your Brev instance:
#   export VLLM_API_KEY="your-secret-token"
#   ./deploy/serve_brev.sh
#
# Then expose port 8000 publicly in Brev (for evaluation handover) or forward via:
#   brev port-forward <instance> --port 8000:8000
set -euo pipefail

MODEL="${OPENAI_MODEL:-Qwen/Qwen3.8-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-${OPENAI_API_KEY:-changeme-sphere-forge}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SPEC_CONFIG="${SPEC_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":3}}"
LIMIT_MM="${LIMIT_MM:-{\"image\":24}}"

if [[ "$API_KEY" == "changeme-sphere-forge" ]]; then
    echo "[warn] Using default VLLM_API_KEY; set VLLM_API_KEY before exposing publicly." >&2
fi

COMMON_ARGS=(
    --model "${MODEL}"
    --api-key "${API_KEY}"
    --reasoning-parser qwen3
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --limit-mm-per-prompt "${LIMIT_MM}"
    --enable-prefix-caching
    --speculative-config "${SPEC_CONFIG}"
)

if command -v vllm >/dev/null 2>&1; then
    echo "Starting native vLLM server for ${MODEL} on ${HOST}:${PORT}..."
    exec vllm serve "${MODEL}" --host "${HOST}" --port "${PORT}" \
        --api-key "${API_KEY}" \
        --reasoning-parser qwen3 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --limit-mm-per-prompt "${LIMIT_MM}" \
        --enable-prefix-caching \
        --speculative-config "${SPEC_CONFIG}"
else
    echo "Starting Dockerized vLLM server for ${MODEL} on ${HOST}:${PORT}..."
    mkdir -p "${HOME}/hf"
    docker rm -f vllm >/dev/null 2>&1 || true
    exec docker run --rm --name vllm --gpus all --ipc=host -p "${PORT}:8000" \
        -v "${HOME}/hf:/root/.cache/huggingface" \
        -e "VLLM_API_KEY=${API_KEY}" \
        vllm/vllm-openai:latest \
        "${COMMON_ARGS[@]}"
fi
