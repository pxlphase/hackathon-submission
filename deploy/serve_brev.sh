#!/usr/bin/env bash
# Launch Qwen/Qwen3.8-27B-FP8 (or Qwen/Qwen3.8-27B) on NVIDIA Brev (1x RTX PRO 6000 96 GB)
# with qwen3 reasoning parser, qwen3_coder tool-calling parser, prefix caching, and safe headroom.
#
# Usage on your Brev instance:
#   export VLLM_API_KEY="your-secret-token"
#   ./deploy/serve_brev.sh        # foreground
#   ./deploy/serve_brev.sh --bg   # background + wait until /v1/models is healthy
#
# Optional model overrides:
#   VLLM_MODEL=Qwen/Qwen3.8-27B ./deploy/serve_brev.sh --bg       # Full BF16 (~55 GB)
#   VLLM_MODEL=Inferact/Qwen3.8-27B-NVFP4 ./deploy/serve_brev.sh --bg
set -euo pipefail

BG_MODE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bg)
            BG_MODE=1
            shift
            ;;
        --fg)
            BG_MODE=0
            shift
            ;;
        *)
            echo "[error] Unknown argument: $1 (supported: --bg | --fg)" >&2
            exit 2
            ;;
    esac
done

MODEL="${VLLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-${OPENAI_API_KEY:-changeme-sphere-forge}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
SPEC_CONFIG="${SPEC_CONFIG:-}"
LIMIT_MM="${LIMIT_MM:-{\"image\":8}}"
LOG_FILE="${VLLM_LOG_FILE:-/tmp/vllm.log}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-240}"

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

export PYTHONUNBUFFERED=1

wait_for_vllm() {
    local pid="$1"
    local is_docker="$2"
    local url="http://127.0.0.1:${PORT}/v1/models"
    local deadline=$((SECONDS + WAIT_TIMEOUT_S))
    echo "Waiting for vLLM at ${url} (timeout=${WAIT_TIMEOUT_S}s, log=${LOG_FILE})..."
    while (( SECONDS < deadline )); do
        if curl -fsS -H "Authorization: Bearer ${API_KEY}" "${url}" >/tmp/.vllm_models_resp 2>/dev/null; then
            echo "vLLM ready! Served models:"
            cat /tmp/.vllm_models_resp
            echo
            rm -f /tmp/.vllm_models_resp
            return 0
        fi
        if [[ "${is_docker}" == "0" ]] && ! kill -0 "${pid}" 2>/dev/null; then
            echo "[error] vLLM server process (pid=${pid}) exited unexpectedly. Last 50 lines of ${LOG_FILE}:" >&2
            tail -n 50 "${LOG_FILE}" >&2 || true
            return 1
        fi
        if [[ "${is_docker}" == "1" ]] && ! docker inspect -f '{{.State.Running}}' vllm 2>/dev/null | grep -q true; then
            echo "[error] Docker container 'vllm' exited unexpectedly. Last 50 log lines:" >&2
            docker logs --tail 50 vllm >&2 || true
            return 1
        fi
        sleep 2
    done
    echo "[error] Timed out after ${WAIT_TIMEOUT_S}s waiting for ${url}. Tail of logs:" >&2
    if [[ "${is_docker}" == "1" ]]; then
        docker logs --tail 50 vllm >&2 || true
    else
        tail -n 50 "${LOG_FILE}" >&2 || true
    fi
    return 1
}

if command -v vllm >/dev/null 2>&1; then
    echo "Starting native vLLM server for ${MODEL} (served as ${SERVED_MODEL_NAME}) on ${HOST}:${PORT}..."
    if [[ "${BG_MODE}" == "1" ]]; then
        nohup vllm serve "${MODEL}" "${SERVE_FLAGS[@]}" > "${LOG_FILE}" 2>&1 < /dev/null &
        VLLM_PID=$!
        echo "vLLM launched in background (pid=${VLLM_PID}, log=${LOG_FILE})."
        wait_for_vllm "${VLLM_PID}" 0
    else
        exec vllm serve "${MODEL}" "${SERVE_FLAGS[@]}"
    fi
else
    echo "Starting Dockerized vLLM server for ${MODEL} (served as ${SERVED_MODEL_NAME}) on ${HOST}:${PORT}..."
    mkdir -p "${HOME}/hf"
    docker rm -f vllm >/dev/null 2>&1 || true
    if [[ "${BG_MODE}" == "1" ]]; then
        docker run -d --rm --name vllm --gpus all --ipc=host -p "${PORT}:8000" \
            -v "${HOME}/hf:/root/.cache/huggingface" \
            -e "VLLM_API_KEY=${API_KEY}" \
            vllm/vllm-openai:latest \
            --model "${MODEL}" \
            "${SERVE_FLAGS[@]}" >/dev/null
        wait_for_vllm "0" 1
    else
        exec docker run --rm --name vllm --gpus all --ipc=host -p "${PORT}:8000" \
            -v "${HOME}/hf:/root/.cache/huggingface" \
            -e "VLLM_API_KEY=${API_KEY}" \
            vllm/vllm-openai:latest \
            --model "${MODEL}" \
            "${SERVE_FLAGS[@]}"
    fi
fi
