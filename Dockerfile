FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /uvx /bin/

# Install minimal headless shared libraries linked by cadquery-ocp (OpenCASCADE)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libexpat1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib

# Install dependencies (cached layer, only rebuilt when the lockfile changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy the project into the image
COPY . /app

# Install the project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# Non-secret agent runtime defaults (README: Configuration).
# Endpoint credentials are injected via <team>.env at code freeze, or set here for dev builds.
ENV CAD_GLOBAL_TIMEOUT_S=3300 \
    CAD_MAX_WORKERS=4 \
    CAD_IOU_EARLY_EXIT=0.95 \
    CAD_THINKING=auto

# ENV OPENAI_BASE_URL=http://<your Brev instance>:8000/v1 \
#     OPENAI_API_KEY=<your vLLM --api-key> \
#     OPENAI_MODEL=Qwen/Qwen3.8-27B

CMD ["python", "scripts/main.py"]
