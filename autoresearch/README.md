# Autoresearch Harness for Parameter Golf

An autonomous experiment harness inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), adapted for the [Parameter Golf](https://github.com/openai/parameter-golf) challenge.

## Key Differences from Vanilla Autoresearch

| Aspect | karpathy/autoresearch | This harness |
|--------|----------------------|--------------|
| Agent freedom | Free-form Python edits | Constrained JSON mutations |
| Scoring | Single metric (val_bpb) | Post-quant BPB + size check |
| Validation | Single stage (5 min) | Staged: proxy → full → promoted |
| Constraints | None | 16MB artifact, param bounds |
| Code changes | Direct file edits | Pre-vetted patch system |
| Dedup | None | Config hash deduplication |

## Architecture

```
autoresearch/
├── program.md              # Agent instructions (structured experiment protocol)
├── experiment_schema.py    # Experiment definition, bounds, validation, promotion rules
├── autoresearch_golf.py    # Main orchestration loop
├── run_experiment.sh       # Fixed training runner (not agent-editable)
├── setup_runpod.sh         # One-shot cloud setup
├── Dockerfile              # Containerized deployment
├── results.jsonl           # Experiment results (auto-generated)
├── experiments.jsonl       # Experiment definitions (auto-generated)
└── logs/                   # Per-experiment training logs
```

## Quick Start

### Local (1 GPU)

```bash
# Setup
bash autoresearch/setup_runpod.sh

# Run proxy experiments
export ANTHROPIC_API_KEY='sk-ant-...'
python3 autoresearch/autoresearch_golf.py --max-experiments 50
```

### RunPod (8xH100)

```bash
# On a fresh RunPod pod
git clone <your-fork> && cd parameter-golf
bash autoresearch/setup_runpod.sh

export ANTHROPIC_API_KEY='sk-ant-...'
python3 autoresearch/autoresearch_golf.py \
  --max-experiments 100 \
  --full-ngpu 8 \
  --full-wallclock 600
```

### Docker

```bash
docker build -t pgolf-research -f autoresearch/Dockerfile .
docker run --gpus all -e ANTHROPIC_API_KEY=sk-ant-... \
  -v $(pwd)/results:/workspace/parameter-golf/autoresearch \
  pgolf-research --max-experiments 100
```

## How It Works

1. **Agent proposes** an experiment as a JSON mutation (param overrides + optional code patches)
2. **Harness validates** against declared parameter bounds and constraint rules
3. **Harness runs** proxy training (~25s on 1 GPU)
4. **Harness scores** using post-quantization val_bpb (the challenge metric)
5. **Promotion check**: if BPB improved by >= 0.003, queue for full 10-min 8xH100 validation
6. **Agent receives** structured results and proposes the next experiment

## Extending

### Adding new code patches

Edit `AVAILABLE_PATCHES` in `experiment_schema.py`. Each patch needs:
- A unique ID
- A description
- List of conflicting patches
- Any implicit parameter adjustments

The actual code changes for patches should be implemented as conditional branches
in `train_gpt.py` controlled by environment variables.

### Adding new tunable parameters

Add bounds to `PARAM_BOUNDS` in `experiment_schema.py`. The parameter name must
match an environment variable that `train_gpt.py` reads.
