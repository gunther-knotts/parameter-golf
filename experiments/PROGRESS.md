# Parameter Golf: Experiment Log & Strategy

## Challenge Overview

**Goal**: Train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100s. Evaluated by bits-per-byte (BPB) on FineWeb validation — a tokenizer-agnostic compression metric.

**Current leaderboard SOTA**: 1.2244 BPB (naive baseline)

## Baseline Analysis

### Architecture
- 9-layer transformer, 512 dim, 1024 vocab, tied embeddings
- 8 attention heads, 4 KV heads (GQA), relu² MLP (2x expansion)
- U-Net skip connections (encoder-decoder with learned skip weights)
- x0 residual mixing (shortcut from embedding to every layer)
- Muon optimizer for matrix params, Adam for scalars/embeddings
- 17.06M parameters, compresses to 15.86MB (int8 + zlib)

### Key Numbers (H100 Baseline)

| Metric | Value |
|---|---|
| Post-quant BPB | 1.2244 |
| Pre-quant BPB | 1.2172 |
| Quantization gap | 0.0072 |
| Steps completed (10 min) | 13,780 |
| Step time | ~43.5ms |
| Tokens seen | ~7.2B |

### The 4-Hour Run Tells the Whole Story

The same architecture trained for 4 hours reaches 1.1749 pre-quant BPB — but after int8 quantization it only scores 1.2074. The quantization gap balloons from 0.007 to **0.033** with more training. This means:

1. The architecture has significant capacity headroom (it CAN reach 1.175)
2. Quantization robustness is a first-order concern — it erases most of the gains from longer training
3. Improvements that are robust to quantization will compound better than raw loss improvements

## Experiment Results

### Local Protocol
- 500 steps, TRAIN_BATCH_TOKENS=32768, Apple Silicon MLX
- Fixed seed 1337, validation every 100 steps
- ~50 min per run (training + validation sweeps)

### Exp 0: Baseline Reference (Local)

| Step | val_bpb |
|------|---------|
| 100 | 2.298 |
| 200 | 2.056 |
| 300 | 1.907 |
| 400 | 1.801 |
| 500 | 1.740 |

- Post-quant BPB: 1.7444
- Quantization gap: 0.0045
- Compressed size: 9.76MB

### Exp 1: SwiGLU Activation (Local) — WIN

Replaced relu² MLP (`proj(relu(fc(x))²)`) with SwiGLU (`down(silu(gate(x)) * up(x))`). Hidden dim reduced from 1024 to 682 for parameter parity (3 matrices instead of 2).

| Step | Baseline BPB | SwiGLU BPB | Delta |
|------|-------------|------------|-------|
| 100 | 2.298 | 2.283 | -0.015 |
| 200 | 2.056 | 2.027 | -0.030 |
| 300 | 1.907 | 1.852 | -0.056 |
| 400 | 1.801 | 1.756 | -0.045 |
| 500 | 1.740 | **1.701** | **-0.039** |

- Post-quant BPB: **1.7054** (vs 1.7444 baseline)
- Quantization gap: 0.0040 (slightly better than baseline)
- Compressed size: 10.04MB
- **Verdict: Clear win. 0.039 BPB improvement, 8x the significance threshold.**

## Next Steps (Immediate)

### Exp 2: QAT with Straight-Through Estimator
- Simulate int8 rounding in the forward pass during training
- Backprop through the rounding via straight-through estimator
- Start QAT after step 100 warmup
- Target: shrink the quantization gap without hurting pre-quant BPB
- **Status: Ready to implement**

### Exp 3: SwiGLU + QAT Combined
- Apply both changes together, verify they compose
- **Status: Blocked on Exp 2 results**

### Validate on H100 (RunPod)
- Port SwiGLU to CUDA script (already done in train_gpt.py)
- Run baseline + SwiGLU on 1xH100 to confirm improvement transfers
- **Status: RunPod account set up, ready to go**

## Full Improvement Ideas (Prioritized)

### Tier 1: High Impact, Well-Understood

**1. SwiGLU Activation** — CONFIRMED WIN locally. Port to H100 for full-scale validation.

**2. QAT (Quantization-Aware Training)** — The 4-hour run proves quantization costs 0.033 BPB at convergence. Straight-through estimator or noise injection during training could recover most of this. Even the 10-min baseline loses 0.007. This is essentially free BPB.

**3. Depth Recurrence / Parameter Tying** — Share weights across layers. E.g., 4 unique blocks looped 3x = 12 effective layers for 4 blocks of parameters. Frees ~10M parameters for wider model or larger vocab. The README explicitly lists this as an intended exploration direction. Risk: slower throughput from more sequential computation.

### Tier 2: Medium Impact, Needs Experimentation

**4. Vocabulary Size Optimization** — The metric is bits per BYTE, not per token. BPB = (loss/ln2) × (tokens/bytes). Larger vocab means more bytes per token (favorable ratio), but harder per-token prediction and more embedding parameters. The sweet spot might be 2048 or 4096 instead of 1024. Data pipeline supports `sp4096` and `byte260` variants.

**5. Learning Rate Schedule** — Currently flat LR with linear warmdown in the last ~1200 steps. Cosine decay, warm restarts, or different warmdown fractions could help. The warmdown consumes ~9% of the wall-clock budget.

**6. Low-Rank Factorization** — Factor weight matrices as W = UV where U is d×r and V is r×d. More layers or wider models within the same parameter budget at the cost of expressiveness per layer.

### Tier 3: High Ceiling, Speculative

**7. Test-Time Training (TTT)** — The rules allow 10 additional minutes on 8xH100 for evaluation. The baseline uses 1.4 seconds. Online adaptation during eval — taking gradient steps on each document before predicting subsequent tokens — could dramatically improve BPB without changing the model artifact at all.

**8. Longer Eval Context** — Train at seq_len=1024 but evaluate at much longer sequences using RoPE scaling (NTK-aware or YaRN). More context = better predictions on later tokens.

**9. Low-Bit Training (BitNet-style)** — Train directly in 1.58-bit (ternary) or 2-bit. Radically reduces model size, potentially allowing 4-8x more parameters within 16MB. A 68M-parameter ternary model would massively outscale the current 17M fp32-trained model.

**10. Meta-Learning for TTT** — Train the model specifically to be good at online adaptation. The training objective includes a "how well does this model adapt at test time" signal. Combines ideas 7 and 9.

### Tier 4: Incremental Tuning

**11. GQA Ratio** — Currently 8:4 (Q:KV heads). Try 8:2 for parameter savings or 8:8 (MHA) for quality.

**12. MLP Expansion Ratio** — Currently 2x. Standard transformers use 4x. With SwiGLU, the optimal ratio at this scale might differ.

**13. Logit Softcap Tuning** — Currently 30.0. With vocab=1024, the effective logit range might not need capping, or the cap could be tuned.

**14. Batch Size Optimization** — 524K tokens/step might not be optimal. Smaller batch = more steps in 10 min = more learning, but noisier gradients.

**15. Training Throughput** — Faster kernels, better overlapping of communication and computation. Every ms/step saved means more training in 10 minutes. The modded-nanogpt community lives here.

## Infrastructure

### Local (Apple Silicon)
- MLX training via `train_gpt_mlx.py`
- ~1000ms/step at 32K batch tokens
- Good for quick architecture validation (relative rankings transfer to H100)
- Dependencies managed via uv

### Cloud (RunPod)
- 1xH100 (~$3/hr) for iteration — `nproc_per_node=1`, ~350ms/step
- 8xH100 (~$20/hr) for final leaderboard validation — matches competition conditions exactly
- Official Parameter Golf template pre-installed with all CUDA deps
- Apply for OpenAI compute credits: https://openai.com/index/parameter-golf/#credit-form

### Experiment Workflow
1. Implement change in MLX script, run locally for quick signal
2. Port to CUDA script, validate on 1xH100
3. Promote winners to 8xH100 full-scale run
4. Submit PR with best combination
