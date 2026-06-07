#!/usr/bin/env bash
# One-time build of llama.cpp's `llama-server` with CUDA, for local Qwen3.6 slogan generation.
# Run when a GPU is free (the build itself is CPU, but verify needs the GPU). ~10-15 min.
#
#   bash scripts/setup_llama_server.sh [DEST_DIR]   # default: ~/llama.cpp
#
# Qwen3.6-35B-A3B has a recent architecture (Gated DeltaNet + MoE); build from up-to-date
# llama.cpp. The Unsloth GGUF in benbi already loads under llama-cpp-python 0.3.23, so a current
# upstream llama.cpp supports it. If the loader ever rejects the arch, check out the tag whose
# ggml matches llama-cpp-python 0.3.23.
set -euo pipefail

DEST="${1:-$HOME/llama.cpp}"
GGUF="/home/krishi/workspace/benbi/models/qwen36_gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

if [ ! -d "$DEST/.git" ]; then
  echo "[setup] cloning llama.cpp -> $DEST"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$DEST"
else
  echo "[setup] updating $DEST"; git -C "$DEST" pull --ff-only || true
fi

echo "[setup] configuring (CUDA)"
cmake -S "$DEST" -B "$DEST/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
echo "[setup] building llama-server"
cmake --build "$DEST/build" --config Release -j --target llama-server

BIN="$DEST/build/bin/llama-server"
echo "[setup] built: $BIN"
echo
echo "Smoke test (loads the model on GPU 0, then exits):"
echo "  CUDA_VISIBLE_DEVICES=0 $BIN -m $GGUF -ngl 999 --parallel 4 -c 8192 --port 8080 &"
echo "  curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "    -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Slogan: every continuous cubic graph splits into a tree, matching, cycles. /no_think\"}],\"max_tokens\":40}'"
