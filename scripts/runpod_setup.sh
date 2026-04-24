#!/usr/bin/env bash
# Bootstrap a fresh Runpod pod for news-translation-engine.
#
# Recommended Runpod template:
#   - PyTorch 2.4+ / CUDA 12.1+ image (runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04 or similar)
#   - Minimum 24GB VRAM for 7B QLoRA; 80GB for 32B inference (silver label gen).
#   - Persistent volume mounted at /workspace is recommended (caches HF models
#     and FLORES tarball across pod restarts).
#
# Usage (on the pod, after `git clone`):
#   bash scripts/runpod_setup.sh [--extra eval|train|data|serve|all]
#
# Env vars (export before running, or put in /workspace/.env):
#   HF_TOKEN           — for gated models (Llama etc.)
#   WANDB_API_KEY      — optional, for experiment tracking
#   FLASH_ATTN=1       — optional, install flash-attn after sync (adds 5–15 min build)

set -euo pipefail

EXTRA="${1:---extra}"
GROUP="${2:-train}"

if [[ "${EXTRA}" != "--extra" ]]; then
  GROUP="train"
fi

echo "=== news-translation-engine Runpod setup ==="
echo "Extras group: ${GROUP}"

# 1. Install uv if missing
if ! command -v uv >/dev/null 2>&1; then
  echo ">>> Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. Redirect HF cache to persistent volume if /workspace exists
if [[ -d /workspace ]]; then
  export HF_HOME="/workspace/.cache/huggingface"
  # Disable the xet downloader — on Runpod it stages reconstruction chunks
  # under the small container overlay and hits EDQUOT (os error 122) mid-download.
  # Regular resolve+download goes straight into HF_HOME on /workspace.
  export HF_HUB_DISABLE_XET=1
  # Disable hf_transfer (multi-threaded accelerator). It masks errors with an
  # opaque "An error occurred while downloading using hf_transfer" and gives
  # no retry granularity. Single-threaded is slower but actually finishes.
  export HF_HUB_ENABLE_HF_TRANSFER=0
  export UV_CACHE_DIR="/workspace/.cache/uv"
  mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}"
  echo ">>> HF_HOME=${HF_HOME}"
  echo ">>> HF_HUB_DISABLE_XET=1"
  echo ">>> HF_HUB_ENABLE_HF_TRANSFER=0"
  echo ">>> UV_CACHE_DIR=${UV_CACHE_DIR}"

  # Persist across shell sessions on the pod
  {
    echo "export HF_HOME=${HF_HOME}"
    echo "export HF_HUB_DISABLE_XET=1"
    echo "export HF_HUB_ENABLE_HF_TRANSFER=0"
    echo "export UV_CACHE_DIR=${UV_CACHE_DIR}"
    echo 'export PATH="$HOME/.local/bin:$PATH"'
  } >>"$HOME/.bashrc"
fi

# 3. Load .env if present
if [[ -f .env ]]; then
  echo ">>> Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# 4. Install deps
if [[ "${GROUP}" == "all" ]]; then
  uv sync --extra eval --extra train --extra data --extra serve
else
  uv sync --extra "${GROUP}"
fi

# 4b. Optional flash-attn install (requires torch already in venv — so run after sync)
if [[ "${FLASH_ATTN:-0}" == "1" ]]; then
  echo ">>> Installing flash-attn (this compiles a wheel, ~5-15 min)..."
  uv pip install flash-attn --no-build-isolation
fi

# 5. GPU sanity check
if uv run python -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name(0))" 2>/dev/null; then
  echo ">>> GPU detected"
else
  echo ">>> WARN: no GPU detected. Training and COMET eval will fail." >&2
fi

echo "=== Setup complete ==="
echo "Next: uv run python -m evaluation.eval_bleu_comet --model Qwen/Qwen2.5-7B-Instruct --dataset flores --split devtest --direction ko2en --output evaluation/results/qwen2.5-7b_flores_ko2en_baseline.json"
