#!/usr/bin/env bash
# Launch the FastAPI voice-chat server on 0.0.0.0:8000.
# Assumes vLLM is already running (default: http://127.0.0.1:8001/v1).
# Customize via env: VENV, LLM_BASE, LLM_MODEL, WHISPER_MODEL, KOKORO_VOICE.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VENV="${VENV:-$HERE/venv}"
NV="$VENV/lib/python3.12/site-packages/nvidia"
# faster-whisper's bundled CTranslate2 needs CUDA-12 cublas/cudnn at runtime
if [ -d "$NV/cublas/lib" ]; then
  export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec "$VENV/bin/python" server.py
