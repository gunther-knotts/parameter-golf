#!/usr/bin/env bash
# setup_runpod.sh — One-shot setup for a RunPod instance to run autoresearch.
#
# Usage:
#   # On a fresh RunPod pod (1xH100 or 8xH100):
#   bash autoresearch/setup_runpod.sh
#
# This script:
#   1. Installs dependencies (if not already in the RunPod template)
#   2. Downloads the dataset
#   3. Verifies GPU access
#   4. Runs a smoke test
#   5. Prints instructions for starting the research loop

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================"
echo " Parameter Golf — RunPod Setup"
echo "============================================"
echo ""

# ---------------------
# 1. Check we're in the right place
# ---------------------
if [[ ! -f "$REPO_ROOT/train_gpt.py" ]]; then
    echo "ERROR: train_gpt.py not found. Run this from the parameter-golf repo root."
    exit 1
fi
cd "$REPO_ROOT"

# ---------------------
# 2. GPU check
# ---------------------
echo "[1/5] Checking GPU access..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. CUDA GPUs may not be available."
else
    NGPU=$(nvidia-smi -L | wc -l)
    echo "  Found $NGPU GPU(s):"
    nvidia-smi -L
fi
echo ""

# ---------------------
# 3. Install dependencies
# ---------------------
echo "[2/5] Installing Python dependencies..."
pip install -q numpy torch sentencepiece huggingface-hub datasets tqdm anthropic 2>/dev/null || \
pip install numpy torch sentencepiece huggingface-hub datasets tqdm anthropic
echo "  Done."
echo ""

# ---------------------
# 4. Download dataset
# ---------------------
echo "[3/5] Downloading FineWeb dataset (sp1024)..."
DATASET_DIR="$REPO_ROOT/data/datasets/fineweb10B_sp1024"
if [[ -d "$DATASET_DIR" ]] && ls "$DATASET_DIR"/fineweb_train_*.bin &>/dev/null 2>&1; then
    SHARD_COUNT=$(ls "$DATASET_DIR"/fineweb_train_*.bin 2>/dev/null | wc -l)
    echo "  Dataset already exists ($SHARD_COUNT training shards)."
    if [[ "$SHARD_COUNT" -lt 10 ]]; then
        echo "  WARNING: Only $SHARD_COUNT shards found. For full training, download all 80."
        echo "  Run: python3 data/cached_challenge_fineweb.py --variant sp1024"
    fi
else
    # Download a reasonable default: 10 shards for iteration, full 80 for final runs
    SHARDS="${SETUP_TRAIN_SHARDS:-10}"
    echo "  Downloading $SHARDS training shards..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$SHARDS"
fi
echo ""

# ---------------------
# 5. Verify tokenizer
# ---------------------
echo "[4/5] Verifying tokenizer..."
TOKENIZER="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
if [[ -f "$TOKENIZER" ]]; then
    echo "  Tokenizer found: $TOKENIZER"
else
    echo "  ERROR: Tokenizer not found. Dataset download may have failed."
    exit 1
fi
echo ""

# ---------------------
# 6. Smoke test
# ---------------------
echo "[5/5] Running smoke test (5 steps, ~10 seconds)..."
NGPU=${NGPU:-1}
SMOKE_EXIT=0
MAX_WALLCLOCK_SECONDS=30 \
ITERATIONS=5 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1 \
RUN_ID=smoke_test \
torchrun --standalone --nproc_per_node="$NGPU" train_gpt.py 2>&1 | tail -5 || SMOKE_EXIT=$?

if [[ $SMOKE_EXIT -ne 0 ]]; then
    echo ""
    echo "  WARNING: Smoke test failed (exit code $SMOKE_EXIT)."
    echo "  Check GPU drivers and PyTorch installation."
else
    echo "  Smoke test passed!"
fi
echo ""

# ---------------------
# 7. Make scripts executable
# ---------------------
chmod +x "$SCRIPT_DIR/run_experiment.sh"

# ---------------------
# Print instructions
# ---------------------
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "To start the autonomous research loop:"
echo ""
echo "  # Set your Anthropic API key"
echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
echo ""
echo "  # Run proxy experiments on available GPUs (fast iteration)"
echo "  python3 autoresearch/autoresearch_golf.py \\"
echo "    --agent claude \\"
echo "    --max-experiments 100 \\"
echo "    --proxy-wallclock 25"
echo ""
echo "  # To also run full validation when proxy looks promising:"
echo "  python3 autoresearch/autoresearch_golf.py \\"
echo "    --agent claude \\"
echo "    --max-experiments 100 \\"
echo "    --full-ngpu $NGPU \\"
echo "    --full-wallclock 600"
echo ""
echo "  # To resume after interruption:"
echo "  python3 autoresearch/autoresearch_golf.py --agent claude --resume"
echo ""
echo "Results are saved to autoresearch/results.jsonl"
echo "Experiment definitions in autoresearch/experiments.jsonl"
echo "Logs in autoresearch/logs/"
echo ""
echo "For RunPod 8xH100 final submissions:"
echo "  torchrun --standalone --nproc_per_node=8 train_gpt.py"
echo ""
