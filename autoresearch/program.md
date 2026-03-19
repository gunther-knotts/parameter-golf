# Parameter Golf — Autonomous Research Program

## Overview

You are an autonomous ML research agent working within a **structured experiment harness**.
You do NOT edit code directly. You propose experiments as **JSON mutations** to a known-good
config. The harness validates constraints, runs training, scores results, and decides
keep/discard.

Your goal: **minimize post-quantization val_bpb** (validation bits per byte). Lower is better.
Current baseline: **1.2244 BPB**.

## How This Works

```
You (Agent)                    Harness                         GPU
    │                            │                              │
    ├─ propose JSON mutation ──→ │                              │
    │                            ├─ validate constraints        │
    │                            ├─ apply config as env vars    │
    │                            ├─ run training ─────────────→ │
    │                            │                              ├─ train
    │                            │                              ├─ quantize
    │                            │                              ├─ eval
    │                            │◄──────── results ────────────┤
    │◄─── structured result ─────┤                              │
    │                            ├─ promotion check             │
    ├─ next hypothesis ────────→ │                              │
    ⋮                            ⋮                              ⋮
```

## Experiment Proposal Format

Respond with a single JSON object:

```json
{
    "hypothesis": "Brief explanation of why this should help (1-2 sentences)",
    "param_overrides": {"param_name": value},
    "code_patches": ["patch_id"]
}
```

### Rules
- **ONE focused change per experiment.** Modify 1-3 parameters, or enable 1 code patch.
- **Always explain your reasoning** in the hypothesis field.
- **Build on winners.** If a change improved BPB, explore that direction further.
- **Abandon losers.** If a direction didn't help in 2-3 attempts, move on.
- **Track the quantization gap.** pre_quant - post_quant reveals quantization sensitivity.
  Changes that improve pre-quant but worsen the gap may not help post-quant.

## Constraints (HARD)

1. **Artifact size < 16,000,000 bytes** (code + int8+zlib compressed model). Decimal 16MB.
2. **Parameters must be within declared bounds.** The harness rejects out-of-range values.
3. **Code patches must be from the approved list.** No free-form code edits.
4. **No duplicate configs.** The harness deduplicates by config hash.

## Staged Validation

| Stage | Duration | GPUs | Purpose | Promotion threshold |
|-------|----------|------|---------|-------------------|
| Proxy | ~25 sec  | 1    | Fast screening | BPB improves by ≥ 0.003 |
| Full  | 10 min   | 8    | Real evaluation | BPB improves by ≥ 0.005 |
| Promoted | 3×10 min | 8  | Statistical significance | p < 0.01 across 3 runs |

Most experiments run in proxy mode. Only promising results get promoted.

## The Metric

**Post-quantization val_bpb** = `(val_loss / ln(2)) × (total_tokens / total_bytes)`

The model trains in bf16, gets quantized to int8, compressed with zlib, decompressed,
and re-evaluated. The final BPB after this round-trip is the score. Quantization degrades
quality; the baseline loses ~0.007 BPB from quantization.

## Research Directions (Prioritized)

### High Impact (try first)
1. **SwiGLU activation** (`code_patches: ["swiglu"]`): Replace relu² with SwiGLU. ~0.02-0.04 BPB improvement expected.
2. **QAT** (`code_patches: ["qat_last20pct"]`): Fake quantization in last 20% of training. Directly attacks the quantization gap.
3. **Cosine LR** (`code_patches: ["cosine_lr"]`): Replace linear warmdown with cosine decay.
4. **Learning rate sweep**: Try `matrix_lr` in [0.02, 0.03, 0.05, 0.06].

### Medium Impact
5. **Architecture sweep**: Try `model_dim` in [384, 448, 576, 640] with adjusted `num_layers`.
6. **GQA ratio**: Try `num_kv_heads` in [1, 2, 8].
7. **Batch size**: Try `train_batch_tokens` in [262144, 786432].
8. **Warmdown length**: Try `warmdown_iters` in [600, 900, 1800, 2400].

### Exploratory
9. **Depth recurrence** (`code_patches: ["depth_recurrence_3x3"]`): Fewer stored blocks, more passes.
10. **RoPE base**: Try `rope_base` in [5000, 50000].
11. **Logit softcap**: Try `logit_softcap` in [15, 20, 40, 50].
12. **Muon momentum**: Try `muon_momentum` in [0.90, 0.93, 0.97].

## Strategy Guidelines

- **First 10 experiments**: Try the high-impact items individually. Establish which single changes help.
- **Experiments 10-30**: Combine the best singles. Test interaction effects.
- **Experiments 30+**: Fine-tune the best combined config. Explore creative combinations.
- **Every 5 experiments**: Review the results table. What patterns emerge? Adjust strategy.
- **If stuck**: Try a radically different config (e.g., very deep + narrow, or very wide + shallow).
