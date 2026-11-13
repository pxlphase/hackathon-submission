#!/usr/bin/env bash
# Launch Qwen/Qwen3.8-27B-FP8 on NVIDIA Brev (1x RTX PRO 6000 96 GB)
# with native FP8 weights (~29 GB), FP8 KV cache, MTP speculative decoding,
# qwen3 reasoning parser, qwen3_coder tool-calling parser, and ~46 GB free VRAM headroom.
#
# Usage on your Brev instance:
#   export VLLM_API_KEY="your-secret-token"
#   ./deploy/serve_brev.sh
#
# Optional model overrides:
#   VLLM_MODEL=Inferact/Qwen3.8-27B-NVFP4 ./deploy/serve_brev.sh   # Blackwell W4A4 (~25 GB total)
#   VLLM_MODEL=Qwen/Qwen3.8-27B ./deploy/serve_brev.sh             # Full BF16 (~52 GB weights)
#
# Then expose port 8000 publicly in Brev (for evaluation handover) or forward via:
#   brev port-forward <instance> --port 8000:8000
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-${OPENAI_API_KEY:-changeme-sphere-forge}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-24}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.52}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
SPEC_CONFIG="${SPEC_CONFIG:-{\"method\":\"mtp\",\"num_speculative_tokens\":3}}"
LIMIT_MM="${LIMIT_MM:-{\"image\":24}}"

if [[ "$API_KEY" == "changeme-sphere-forge" ]]; then
    echo "[warn] Using default VLLM_API_KEY; set VLLM_API_KEY before exposing publicly." >&2
fi

SERVE_FLAGS=(
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --api-key "${API_KEY}"
    --reasoning-parser "${REASONING_PARSER}"
    --enable-auto-tool-choice
    --tool-call-parser "${TOOL_CALL_PARSER}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --limit-mm-per-prompt "${LIMIT_MM}"
    --enable-prefix-caching
)
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
    SERVE_FLAGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${SPEC_CONFIG}" ]]; then
    SERVE_FLAGS+=(--speculative-config "${SPEC_CONFIG}")
fi

if command -v vllm >/dev/null 2>&1; then
    echo "Starting native vLLM server for ${MODEL} (served as ${SERVED_MODEL_NAME}) on ${HOST}:${PORT}..."
    exec vllm serve "${MODEL}" "${SERVE_FLAGS[@]}"
else
    echo "Starting Dockerized vLLM server for ${MODEL} (served as ${SERVED_MODEL_NAME}) on ${HOST}:${PORT}..."
    mkdir -p "${HOME}/hf"
    docker rm -f vllm >/dev/null 2>&1 || true
    exec docker run --rm --name vllm --gpus all --ipc=host -p "${PORT}:8000" \
        -v "${HOME}/hf:/root/.cache/huggingface" \
        -e "VLLM_API_KEY=${API_KEY}" \
        vllm/vllm-openai:latest \
        --model "${MODEL}" \
        "${SERVE_FLAGS[@]}"
fi
