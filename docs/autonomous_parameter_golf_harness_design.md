# Autonomous Parameter Golf Harness Design

## Purpose

This document proposes an autonomous research harness for the Parameter Golf repository. The goal is to create a system that can:

- define experiments,
- run them at different fidelity levels,
- parse outcomes automatically,
- rank results using challenge-relevant metrics,
- and recommend or launch next experiments.

The harness should optimize the actual challenge target: post-quantization bits-per-byte (BPB) under the artifact-size and runtime constraints.

## Challenge Constraints That Shape the Harness

The Parameter Golf challenge is unusual in ways that matter for automation:

- The submission artifact is capped at **16,000,000 bytes**, including counted code and compressed model bytes.
- Leaderboard training must run in **under 10 minutes on 8xH100**.
- Evaluation gets a **separate 10-minute budget on 8xH100**.
- No external downloads, training dataset access, or network calls are allowed during evaluation.
- The challenge metric is tokenizer-agnostic **BPB**, so post-training quantization and evaluation policy both matter.

Because of those rules, a naive harness that optimizes only short-run training loss or even pre-quant validation loss will optimize the wrong objective.

## Primary Objective

The harness should optimize the metric already emitted by `train_gpt.py`:

- `final_int8_zlib_roundtrip_exact val_bpb`

This is the best scalar for leaderboard relevance because it measures the round-tripped, compressed artifact after the exact post-training quantization path.

## Secondary Metrics to Track

Every experiment should also record:

- pre-quant `val_loss`
- pre-quant `val_bpb`
- post-quant `val_loss`
- post-quant `val_bpb`
- quantization gap (`postquant_bpb - prequant_bpb`)
- train wallclock
- eval wallclock
- total artifact bytes
- model bytes
- code bytes
- training step count reached
- average step time
- peak memory
- seed
- parent experiment
- hypothesis text

The current training script already logs most of these values, which makes this repo a good fit for an autonomous harness.

## Core Design Principles

### 1. Optimize post-quant performance, not pre-quant performance

The unlimited-compute 4-hour run demonstrates that pre-quant quality can improve dramatically while post-quant quality lags behind. The harness must therefore rank experiments using the exact final post-quant BPB, not raw validation loss.

### 2. Treat evaluation as part of the search space

The challenge allows aggressive evaluation methods and arbitrary evaluation length. The current evaluation path is much cheaper than the available budget, so evaluation policy should be treated as a first-class search dimension.

### 3. Start with constrained autonomy

The current trainer already exposes many useful environment-variable knobs. The first version of the harness should search over:

- environment variables,
- named code variants,
- and a small library of patch templates.

It should not begin with unrestricted code rewriting.

### 4. Use staged fidelity

A single full-fidelity loop would be too expensive and too noisy. The harness should use a promotion ladder:

- cheap smoke/proxy screening,
- medium-fidelity confirmation,
- and full 8xH100 leaderboard-fidelity validation only for the most promising candidates.

### 5. Keep the harness outside the counted artifact path

The challenge counts code bytes in the submission artifact. The harness should therefore live outside any future submission artifact layout and should not bloat the minimal training script used for records.

## System Architecture

### Component 1: Experiment Specification

Each experiment should be defined by a structured spec, for example JSON or YAML, with fields such as:

- `experiment_id`
- `parent_id`
- `created_at`
- `family`
- `hypothesis`
- `mode` (`proxy_local`, `proxy_gpu`, `full_8xh100`, `eval_only`)
- `script_target` (`train_gpt.py`, `train_gpt_mlx.py`)
- `env`
- `patch_variant`
- `seed`
- `replicate_count`
- `notes`

### Component 2: Runner

The runner should:

- materialize environment variables,
- optionally apply a named code variant,
- launch the relevant training or eval command,
- capture logs and artifacts,
- and write run metadata into a results database.

### Component 3: Log Parser

A parser should extract structured metrics from training logs and summary outputs. It should at minimum recover:

- latest pre-quant `val_bpb`
- exact final post-quant `val_bpb`
- artifact bytes
- train and eval times
- peak memory
- step count

The existing trainer log format is already regular enough to support this.

### Component 4: Results Database

Store one row per experiment in SQLite or JSONL. Suggested fields:

- all experiment-spec fields,
- raw parsed metrics,
- derived metrics such as:
  - `quant_gap`
  - `artifact_margin`
  - `score_rank`

This database is the harness memory.

### Component 5: Planner

The planner should:

- inspect results,
- summarize what has and has not worked,
- identify promising directions,
- and propose the next experiment batch.

The first version can be mostly deterministic with optional LLM assistance for hypothesis generation.

### Component 6: Promotion Engine

A promotion engine should move candidates through fidelity levels:

1. proxy screening,
2. medium-fidelity confirmation,
3. full-fidelity training,
4. eval-only optimization.

## Modes of Operation

### Proxy Local

Use `train_gpt_mlx.py` or very small smoke jobs to catch:

- syntax issues,
- shape mismatches,
- and obviously bad ideas.

### Proxy GPU

Use shorter CUDA runs with reduced cost but preserve:

- post-quant scoring,
- log parsing,
- and artifact accounting.

### Full 8xH100

Use the real leaderboard-like path:

- `train_gpt.py`
- 8 GPUs
- 600-second cap
- exact post-quant scoring

### Eval-Only

Run alternate evaluation policies on a fixed trained checkpoint. This mode should search over:

- evaluation sequence length,
- long-context behavior,
- RoPE scaling,
- test-time training,
- and other evaluation-time compute policies.

## Search Space Design

### Tier 1: Environment Variables

The first search tier should use existing exposed knobs such as:

- model shape (`NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`)
- optimizer hyperparameters
- batch tokens
- sequence length
- warmup/warmdown settings
- quantization thresholds and passthrough settings

This provides a large search space without requiring free-form code generation.

### Tier 2: Named Code Variants

Introduce explicit named variants such as:

- `fused_qkv`
- `eval_seq_len_override`
- `rope_eval_scaling`
- `qat_warmdown`
- `recurrent_blocks_4way`
- `recurrent_blocks_3way`
- `control_ttt`

These should be implemented as deliberate branches or templates, not ad hoc one-off edits.

### Tier 3: Open-Ended Codegen

Only after the harness is stable should it be allowed to propose more open-ended structural changes.

## Ranking and Promotion Policy

### Ranking

For full-fidelity experiments, rank by:

1. lowest post-quant BPB,
2. artifact safety margin,
3. evaluation runtime margin,
4. reproducibility across seeds.

### Promotion Rules

Promote if:

- proxy post-quant BPB improves materially,
- or quantization gap shrinks materially,
- or the score is unchanged but bytes or runtime improve.

Reject if:

- only pre-quant metrics improve,
- the artifact size becomes too risky,
- or evaluation cost rises without compensating score gain.

### Replication

For promising candidates, require repeated runs across multiple seeds before trusting small deltas.

## Recommended Search Order

### Family 1: Eval-Only

These should be first because they may deliver the fastest gains:

- long-context eval (`EVAL_SEQ_LEN=2048,4096,8192`)
- streaming eval
- RoPE eval scaling
- control-tensor test-time training

### Family 2: Quantization-Aware Training

These should come next because the post-quant gap appears to be the main bottleneck:

- fake quant during warmdown
- quantization-distortion penalties
- clip-threshold sweeps
- passthrough-threshold sweeps

### Family 3: Throughput and Update Efficiency

Then test:

- batch-token sweeps
- fused QKV
- Muon schedule sweeps

### Family 4: Parameter Tying and Recurrence

After the pipeline is stable, explore:

- 4-way shared recurrent blocks
- 3-way shared recurrent blocks
- widening at matched parameter budget

### Family 5: Tokenizer and Vocabulary

Tokenizer experiments should come later because they require extra correctness scrutiny.

## Failure Modes

The harness should explicitly defend against:

### Goodharting on pre-quant loss

Improved training loss may not improve the final challenge score.

### Proxy mismatch

A change that helps short or cheap runs may not help full 8xH100 runs.

### Metric corruption

Autonomous edits to tokenizer-byte accounting are especially dangerous and should be tightly constrained.

### Code-size drift

Because submission code bytes count toward the artifact cap.

### Invalid evaluation policy

Any eval-time adaptation must remain causally valid and within the challenge rules.

## Suggested Repository Layout

A good first layout would be:

```text
research_harness/
  README.md
  experiment_schema.md
  runner.py
  parse_logs.py
  planner.py
  db.py
  variants/
  configs/
  prompts/
  reports/
```

This keeps the harness separate from the minimal core training script used for records.

## Minimal Trainer Hooks Worth Adding Later

To support the harness cleanly, the most useful future additions to `train_gpt.py` would be:

1. `EVAL_SEQ_LEN` distinct from `TRAIN_SEQ_LEN`
2. optional structured JSON summary output at end of run
3. named eval modes (`standard`, `long_context`, `streaming`)
4. named QAT toggle (`off`, `warmdown`, `full`)
5. named architecture variant toggle

These additions would make the harness much easier to implement without increasing the submission artifact path unnecessarily.

## Immediate Implementation Plan

### Week 1

- define experiment schema
- build log parser
- build results database
- build runner for env-var-only experiments

### Week 2

- add promotion ladder
- add scoreboard and summary reporting
- add `EVAL_SEQ_LEN`
- begin eval-only search

### Week 3

- add QAT warmdown variants
- add throughput variants
- add named code-variant templates

### Week 4

- add recurrent-block variants
- prepare tokenizer/vocab sweep infrastructure
- layer in planner-driven experiment selection

## Conclusion

An autonomous research harness is a strong fit for Parameter Golf, but only if it is challenge-aware. The right design is not a free-form code-writing agent from day one. The right design is a constrained, metrics-first research operating system that:

- ranks by exact post-quant BPB,
- treats evaluation as first-class,
- uses staged fidelity,
- and promotes only the most promising candidates to expensive runs.

That approach should maximize research throughput while minimizing the risk of optimizing the wrong objective.
