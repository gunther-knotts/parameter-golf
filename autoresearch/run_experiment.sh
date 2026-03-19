#!/usr/bin/env bash
# run_experiment.sh — Fixed experiment runner for parameter-golf autoresearch.
# This file should NOT be modified by the research agent.
#
# Usage:
#   bash run_experiment.sh [OPTIONS]
#
# Options:
#   --ngpu N          Number of GPUs (default: auto-detect, fallback 1)
#   --proxy           Proxy mode: short run for fast iteration (~500 steps)
#   --wallclock SECS  Override MAX_WALLCLOCK_SECONDS (default: 600, proxy: 25)
#   --dry-run         Print the command without running
#   --help            Show this help

set -euo pipefail

# ---------------------
# Defaults
# ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_gpt.py"
DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${SCRIPT_DIR}/data/tokenizers/fineweb_1024_bpe.model}"
RUN_ID="${RUN_ID:-autoresearch_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${SCRIPT_DIR}/autoresearch/logs"

NGPU=""
PROXY=0
WALLCLOCK=""
DRY_RUN=0

# ---------------------
# Parse args
# ---------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngpu)    NGPU="$2"; shift 2 ;;
        --proxy)   PROXY=1; shift ;;
        --wallclock) WALLCLOCK="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help)
            head -14 "$0" | tail -12
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------
# Auto-detect GPUs
# ---------------------
if [[ -z "$NGPU" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
        [[ "$NGPU" -lt 1 ]] && NGPU=1
    else
        NGPU=1
    fi
fi

# ---------------------
# Configure mode
# ---------------------
if [[ "$PROXY" -eq 1 ]]; then
    # Proxy mode: fast iteration. ~500 steps at ~43ms/step ≈ 22 seconds.
    # We set a short wallclock and reduce validation frequency.
    WALLCLOCK="${WALLCLOCK:-25}"
    export ITERATIONS="${ITERATIONS:-1000}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-0}"
    export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-131072}"
    export TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-100}"
else
    WALLCLOCK="${WALLCLOCK:-600}"
    export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-1000}"
fi

export MAX_WALLCLOCK_SECONDS="$WALLCLOCK"
export DATA_PATH
export TOKENIZER_PATH
export RUN_ID

# ---------------------
# Prepare log directory
# ---------------------
mkdir -p "$LOG_DIR"
LOGFILE="${LOG_DIR}/${RUN_ID}.log"

# ---------------------
# Build the command
# ---------------------
CMD="torchrun --standalone --nproc_per_node=${NGPU} ${TRAIN_SCRIPT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN — would execute:"
    echo "  $CMD"
    echo "  Environment:"
    echo "    MAX_WALLCLOCK_SECONDS=$WALLCLOCK"
    echo "    DATA_PATH=$DATA_PATH"
    echo "    NGPU=$NGPU"
    echo "    PROXY=$PROXY"
    exit 0
fi

echo "================================================================"
echo "Parameter Golf Experiment: ${RUN_ID}"
echo "  GPUs: ${NGPU}"
echo "  Mode: $([ "$PROXY" -eq 1 ] && echo 'PROXY (~25s)' || echo 'FULL (10 min)')"
echo "  Wallclock: ${WALLCLOCK}s"
echo "  Log: ${LOGFILE}"
echo "================================================================"

# ---------------------
# Run training
# ---------------------
EXIT_CODE=0
$CMD 2>&1 | tee "$LOGFILE" || EXIT_CODE=$?

# ---------------------
# Extract results
# ---------------------
echo ""
echo "================================================================"
echo "EXPERIMENT RESULTS"
echo "================================================================"

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "RESULT: status=crash exit_code=${EXIT_CODE}"
    exit $EXIT_CODE
fi

# Extract post-quant (the real metric)
POST_QUANT_LINE=$(grep "final_int8_zlib_roundtrip val_loss" "$LOGFILE" | tail -1 || true)
if [[ -z "$POST_QUANT_LINE" ]]; then
    echo "RESULT: status=fail reason=no_post_quant_metric"
    exit 1
fi

POST_QUANT_BPB=$(echo "$POST_QUANT_LINE" | grep -oP 'val_bpb:\K[0-9.]+')
POST_QUANT_LOSS=$(echo "$POST_QUANT_LINE" | grep -oP 'val_loss:\K[0-9.]+')

# Extract pre-quant (last validation line before the final roundtrip)
PRE_QUANT_LINE=$(grep "^step:" "$LOGFILE" | grep "val_bpb" | tail -1 || true)
PRE_QUANT_BPB=""
if [[ -n "$PRE_QUANT_LINE" ]]; then
    PRE_QUANT_BPB=$(echo "$PRE_QUANT_LINE" | grep -oP 'val_bpb:\K[0-9.]+')
fi

# Extract model size
SIZE_LINE=$(grep "Total submission size int8" "$LOGFILE" | tail -1 || true)
SIZE_BYTES=""
if [[ -n "$SIZE_LINE" ]]; then
    SIZE_BYTES=$(echo "$SIZE_LINE" | grep -oP '[0-9]+(?= bytes)')
fi

# Extract param count
PARAM_LINE=$(grep "model_params:" "$LOGFILE" | tail -1 || true)
PARAMS=""
if [[ -n "$PARAM_LINE" ]]; then
    PARAMS=$(echo "$PARAM_LINE" | grep -oP 'model_params:\K[0-9]+')
fi

# Extract step count
STEP_LINE=$(grep "stopping_early\|step:" "$LOGFILE" | tail -1 || true)
STEPS=""
if [[ -n "$STEP_LINE" ]]; then
    STEPS=$(echo "$STEP_LINE" | grep -oP 'step:\K[0-9]+')
fi

# Check size constraint
SIZE_OK="ok"
if [[ -n "$SIZE_BYTES" ]] && [[ "$SIZE_BYTES" -ge 16000000 ]]; then
    SIZE_OK="over_budget"
fi

# Print summary
echo "  Post-quant val_bpb : ${POST_QUANT_BPB:-N/A}"
echo "  Post-quant val_loss: ${POST_QUANT_LOSS:-N/A}"
echo "  Pre-quant  val_bpb : ${PRE_QUANT_BPB:-N/A}"
echo "  Quant gap          : $(python3 -c "print(f'{float(\"${POST_QUANT_BPB:-0}\") - float(\"${PRE_QUANT_BPB:-0}\"):.4f}')" 2>/dev/null || echo 'N/A')"
echo "  Model size (bytes) : ${SIZE_BYTES:-N/A}"
echo "  Size budget        : ${SIZE_OK}"
echo "  Parameters         : ${PARAMS:-N/A}"
echo "  Steps completed    : ${STEPS:-N/A}"
echo ""

if [[ "$SIZE_OK" == "over_budget" ]]; then
    echo "RESULT: val_bpb=${POST_QUANT_BPB} size=${SIZE_BYTES} status=over_budget"
else
    echo "RESULT: val_bpb=${POST_QUANT_BPB} size=${SIZE_BYTES} status=ok"
fi
