"""
experiment_schema.py — Defines the structured experiment space for parameter-golf.

Instead of free-form code edits, experiments are defined as structured mutations
to a known-good config. The agent proposes experiments via JSON, the harness
validates constraints, applies the mutation, runs training, and scores.

This file is NOT modified by the agent.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import hashlib
from pathlib import Path
from typing import Any

# ---------------------
# Baseline config: the known-good starting point
# ---------------------

BASELINE_CONFIG = {
    # Model architecture
    "num_layers": 9,
    "model_dim": 512,
    "num_heads": 8,
    "num_kv_heads": 4,
    "mlp_mult": 2,
    "vocab_size": 1024,
    "train_seq_len": 1024,
    "tie_embeddings": 1,
    "rope_base": 10000.0,
    "logit_softcap": 30.0,
    "qk_gain_init": 1.5,

    # Optimizer
    "matrix_lr": 0.04,
    "scalar_lr": 0.04,
    "tied_embed_lr": 0.05,
    "embed_lr": 0.6,
    "head_lr": 0.008,
    "muon_momentum": 0.95,
    "muon_backend_steps": 5,
    "muon_momentum_warmup_start": 0.85,
    "muon_momentum_warmup_steps": 500,
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_eps": 1e-8,
    "grad_clip_norm": 0.0,

    # Training
    "train_batch_tokens": 524288,
    "warmdown_iters": 1200,
    "iterations": 20000,

    # Code patches (for architectural changes that can't be expressed as env vars)
    "code_patches": [],
}

# ---------------------
# Allowed mutation ranges — the agent can ONLY set values within these bounds
# ---------------------

PARAM_BOUNDS = {
    # Architecture
    "num_layers":       {"type": "int",   "min": 4,     "max": 18},
    "model_dim":        {"type": "int",   "min": 256,   "max": 1024,  "step": 64},
    "num_heads":        {"type": "int",   "min": 4,     "max": 16},
    "num_kv_heads":     {"type": "int",   "min": 1,     "max": 16},
    "mlp_mult":         {"type": "int",   "min": 1,     "max": 4},
    "vocab_size":       {"type": "choice", "values": [512, 1024, 2048, 4096, 8192]},
    "train_seq_len":    {"type": "choice", "values": [256, 512, 1024, 2048]},
    "tie_embeddings":   {"type": "choice", "values": [0, 1]},
    "rope_base":        {"type": "float", "min": 1000.0, "max": 1000000.0},
    "logit_softcap":    {"type": "float", "min": 5.0,    "max": 100.0},
    "qk_gain_init":     {"type": "float", "min": 0.5,    "max": 3.0},

    # Optimizer
    "matrix_lr":        {"type": "float", "min": 0.005,  "max": 0.2},
    "scalar_lr":        {"type": "float", "min": 0.005,  "max": 0.2},
    "tied_embed_lr":    {"type": "float", "min": 0.005,  "max": 0.2},
    "embed_lr":         {"type": "float", "min": 0.05,   "max": 2.0},
    "head_lr":          {"type": "float", "min": 0.001,  "max": 0.1},
    "muon_momentum":    {"type": "float", "min": 0.8,    "max": 0.99},
    "muon_backend_steps": {"type": "int", "min": 3,      "max": 10},
    "muon_momentum_warmup_start": {"type": "float", "min": 0.5, "max": 0.95},
    "muon_momentum_warmup_steps": {"type": "int", "min": 0, "max": 2000},
    "beta1":            {"type": "float", "min": 0.8,    "max": 0.99},
    "beta2":            {"type": "float", "min": 0.9,    "max": 0.999},
    "grad_clip_norm":   {"type": "float", "min": 0.0,    "max": 10.0},

    # Training
    "train_batch_tokens": {"type": "choice", "values": [131072, 262144, 524288, 786432, 1048576]},
    "warmdown_iters":   {"type": "int",   "min": 200,   "max": 5000},
    "iterations":       {"type": "int",   "min": 5000,  "max": 50000},
}

# ---------------------
# Code patches: predefined, vetted architectural changes
# ---------------------
# The agent can enable these by name. They modify train_gpt.py in controlled ways.
# Each patch has a unique ID, a description, and the actual code diff.

AVAILABLE_PATCHES = {
    "swiglu": {
        "description": "Replace relu^2 MLP with SwiGLU at parameter parity",
        "conflicts_with": [],
        "param_adjustments": {},  # e.g., hidden dim recalculation handled in patch
    },
    "cosine_lr": {
        "description": "Replace linear warmdown with cosine decay + warmup",
        "conflicts_with": [],
        "param_adjustments": {},
    },
    "qat_last20pct": {
        "description": "Quantization-aware training in last 20% of steps",
        "conflicts_with": [],
        "param_adjustments": {},
    },
    "weight_quant_reg": {
        "description": "L2 regularization toward int8 grid in last 20% of steps",
        "conflicts_with": ["qat_last20pct"],  # pick one or the other
        "param_adjustments": {},
    },
    "depth_recurrence_3x3": {
        "description": "3 unique blocks × 3 passes = 9 effective layers",
        "conflicts_with": ["depth_recurrence_2x5"],
        "param_adjustments": {"num_layers": 3},  # 3 stored, 9 effective
    },
    "depth_recurrence_2x5": {
        "description": "2 unique blocks × 5 passes = 10 effective layers",
        "conflicts_with": ["depth_recurrence_3x3"],
        "param_adjustments": {"num_layers": 2},
    },
}


# ---------------------
# Experiment definition
# ---------------------

@dataclasses.dataclass
class Experiment:
    """A single proposed experiment."""
    experiment_id: str
    parent_id: str | None  # which experiment this builds on
    hypothesis: str  # what we expect and why
    param_overrides: dict[str, Any]  # env var overrides vs parent config
    code_patches: list[str]  # list of patch IDs from AVAILABLE_PATCHES
    stage: str = "proxy"  # proxy | full | promoted

    def config_hash(self) -> str:
        """Deterministic hash of the effective config for dedup."""
        effective = self.effective_config()
        raw = json.dumps(effective, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def effective_config(self, parent_config: dict | None = None) -> dict:
        """Compute the full config by applying overrides to parent."""
        base = copy.deepcopy(parent_config or BASELINE_CONFIG)
        base.update(self.param_overrides)
        base["code_patches"] = sorted(self.code_patches)
        # Apply param adjustments from patches
        for patch_id in self.code_patches:
            if patch_id in AVAILABLE_PATCHES:
                base.update(AVAILABLE_PATCHES[patch_id]["param_adjustments"])
        return base

    def to_env_vars(self, parent_config: dict | None = None) -> dict[str, str]:
        """Convert to environment variables for train_gpt.py."""
        config = self.effective_config(parent_config)
        env = {}
        for key, value in config.items():
            if key == "code_patches":
                continue
            env[key.upper()] = str(value)
        return env


@dataclasses.dataclass
class ExperimentResult:
    """Result of running an experiment."""
    experiment_id: str
    pre_quant_bpb: float | None
    post_quant_bpb: float | None
    size_bytes: int | None
    params: int | None
    steps: int | None
    status: str  # ok | crash | over_budget | timeout
    stage: str  # proxy | full
    error_msg: str = ""

    @property
    def is_valid(self) -> bool:
        return (
            self.status == "ok"
            and self.post_quant_bpb is not None
            and self.size_bytes is not None
            and self.size_bytes < 16_000_000
        )

    @property
    def quant_gap(self) -> float | None:
        if self.pre_quant_bpb is not None and self.post_quant_bpb is not None:
            return self.post_quant_bpb - self.pre_quant_bpb
        return None


# ---------------------
# Validation
# ---------------------

def validate_experiment(exp: Experiment, parent_config: dict | None = None) -> list[str]:
    """Validate an experiment proposal. Returns list of error messages (empty = valid)."""
    errors = []
    config = exp.effective_config(parent_config)

    # Check param bounds
    for key, value in exp.param_overrides.items():
        if key not in PARAM_BOUNDS:
            errors.append(f"Unknown parameter: {key}")
            continue
        bounds = PARAM_BOUNDS[key]
        if bounds["type"] == "choice":
            if value not in bounds["values"]:
                errors.append(f"{key}={value} not in allowed values {bounds['values']}")
        elif bounds["type"] == "int":
            if not isinstance(value, int):
                errors.append(f"{key} must be int, got {type(value).__name__}")
            elif value < bounds["min"] or value > bounds["max"]:
                errors.append(f"{key}={value} out of range [{bounds['min']}, {bounds['max']}]")
            elif "step" in bounds and (value - bounds["min"]) % bounds["step"] != 0:
                errors.append(f"{key}={value} must be aligned to step={bounds['step']}")
        elif bounds["type"] == "float":
            if value < bounds["min"] or value > bounds["max"]:
                errors.append(f"{key}={value} out of range [{bounds['min']}, {bounds['max']}]")

    # Check patch conflicts
    for patch_id in exp.code_patches:
        if patch_id not in AVAILABLE_PATCHES:
            errors.append(f"Unknown patch: {patch_id}")
            continue
        patch = AVAILABLE_PATCHES[patch_id]
        for conflict in patch["conflicts_with"]:
            if conflict in exp.code_patches:
                errors.append(f"Patch '{patch_id}' conflicts with '{conflict}'")

    # Sanity checks
    if "num_heads" in config and "model_dim" in config:
        if config["model_dim"] % config["num_heads"] != 0:
            errors.append(f"model_dim={config['model_dim']} not divisible by num_heads={config['num_heads']}")
    if "num_heads" in config and "num_kv_heads" in config:
        if config["num_heads"] % config["num_kv_heads"] != 0:
            errors.append(f"num_heads={config['num_heads']} not divisible by num_kv_heads={config['num_kv_heads']}")

    return errors


# ---------------------
# Promotion rules
# ---------------------

# Stage 1 (proxy): ~25s on 1xGPU, ~500 steps
# Stage 2 (full): 600s on 8xH100, ~13,780 steps
# Promoted: Passes multiple full runs for statistical significance

PROMOTION_RULES = {
    "proxy_to_full": {
        "min_improvement_bpb": 0.003,  # must beat current best by at least this
        "max_size_bytes": 16_000_000,
        "required_status": "ok",
    },
    "full_to_promoted": {
        "min_improvement_bpb": 0.005,  # challenge threshold
        "min_runs": 3,                 # need 3 runs for significance
        "max_p_value": 0.01,           # p < 0.01 required
        "max_size_bytes": 16_000_000,
    },
}

def should_promote(result: ExperimentResult, current_best_bpb: float, stage: str) -> bool:
    """Decide if an experiment should be promoted to the next stage."""
    if not result.is_valid:
        return False
    rules = PROMOTION_RULES.get(f"{stage}_to_{'full' if stage == 'proxy' else 'promoted'}")
    if rules is None:
        return False
    improvement = current_best_bpb - result.post_quant_bpb
    return improvement >= rules["min_improvement_bpb"]


# ---------------------
# Serialization
# ---------------------

def experiment_to_dict(exp: Experiment) -> dict:
    return dataclasses.asdict(exp)

def experiment_from_dict(d: dict) -> Experiment:
    return Experiment(**d)

def result_to_dict(r: ExperimentResult) -> dict:
    return dataclasses.asdict(r)

def result_from_dict(d: dict) -> ExperimentResult:
    return ExperimentResult(**d)
