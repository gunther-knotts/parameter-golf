#!/usr/bin/env python3
"""
autoresearch_golf.py — Autonomous experiment harness for parameter-golf.

This is NOT "agent hacks Python freely." This is a structured harness that:
1. Asks an LLM agent to propose experiments as structured JSON mutations
2. Validates proposals against hard constraints
3. Runs training with the proposed config
4. Scores results with the real post-quant BPB metric
5. Decides keep/discard via promotion rules
6. Stages: proxy (fast) → full (10min 8xH100) → promoted (multi-run significance)

Usage:
    # Run with Claude as the agent (requires ANTHROPIC_API_KEY)
    python autoresearch_golf.py --agent claude --max-experiments 100

    # Run with a local JSON file of experiments (no agent, just runner)
    python autoresearch_golf.py --from-file experiments.jsonl

    # Resume from existing results
    python autoresearch_golf.py --agent claude --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from experiment_schema import (
    BASELINE_CONFIG,
    AVAILABLE_PATCHES,
    Experiment,
    ExperimentResult,
    validate_experiment,
    should_promote,
    experiment_to_dict,
    experiment_from_dict,
    result_to_dict,
)

# ---------------------
# Paths
# ---------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_DIR = Path(__file__).resolve().parent
RESULTS_FILE = HARNESS_DIR / "results.jsonl"
EXPERIMENTS_FILE = HARNESS_DIR / "experiments.jsonl"
BEST_CONFIG_FILE = HARNESS_DIR / "best_config.json"
RUN_SCRIPT = HARNESS_DIR / "run_experiment.sh"


# ---------------------
# Agent interface
# ---------------------

AGENT_SYSTEM_PROMPT = """\
You are an autonomous ML research agent optimizing a small language model for the \
Parameter Golf challenge. Your goal: minimize post-quantization val_bpb (bits per byte).

You propose experiments as structured JSON. You do NOT edit code directly. \
The harness validates your proposal, runs training, and reports results.

## Current State
{state_summary}

## Available Parameters (with bounds)
{param_bounds}

## Available Code Patches
{available_patches}

## Recent Experiment History
{recent_history}

## Your Task
Propose the next experiment. Respond with a JSON object:
```json
{{
    "hypothesis": "Why this change should help (1-2 sentences)",
    "param_overrides": {{"param_name": value, ...}},
    "code_patches": ["patch_id", ...]
}}
```

Rules:
- Make ONE focused change per experiment (1-3 params, or 1 patch).
- Explain your reasoning in the hypothesis.
- Build on what worked. Discard what didn't.
- Consider the quantization gap — changes that worsen it may not help post-quant.
- The size budget is 16,000,000 bytes. Wider/deeper models compress to more bytes.
- If you're stuck, try something creative. Varied exploration beats repeated tuning.
"""


def format_state_summary(results: list[dict]) -> str:
    """Summarize the current experiment state for the agent."""
    valid = [r for r in results if r["status"] == "ok" and r.get("post_quant_bpb")]
    if not valid:
        return "No experiments completed yet. Baseline val_bpb is ~1.2244."

    best = min(valid, key=lambda r: r["post_quant_bpb"])
    latest = valid[-1]
    return (
        f"Best post_quant_bpb: {best['post_quant_bpb']:.4f} "
        f"(experiment {best['experiment_id']})\n"
        f"Latest: {latest['post_quant_bpb']:.4f} ({latest['experiment_id']})\n"
        f"Total experiments: {len(results)} "
        f"({sum(1 for r in results if r['status'] == 'ok')} ok, "
        f"{sum(1 for r in results if r['status'] == 'crash')} crashed)\n"
        f"Baseline: 1.2244 BPB"
    )


def format_param_bounds() -> str:
    """Format parameter bounds for the agent prompt."""
    from experiment_schema import PARAM_BOUNDS
    lines = []
    for key, bounds in sorted(PARAM_BOUNDS.items()):
        if bounds["type"] == "choice":
            lines.append(f"  {key}: one of {bounds['values']}")
        else:
            extra = f" (step={bounds['step']})" if "step" in bounds else ""
            lines.append(f"  {key}: {bounds['type']} [{bounds['min']}, {bounds['max']}]{extra}")
    return "\n".join(lines)


def format_patches() -> str:
    """Format available patches for the agent prompt."""
    lines = []
    for pid, info in AVAILABLE_PATCHES.items():
        conflicts = f" (conflicts: {info['conflicts_with']})" if info["conflicts_with"] else ""
        lines.append(f"  {pid}: {info['description']}{conflicts}")
    return "\n".join(lines)


def format_history(results: list[dict], n: int = 15) -> str:
    """Format recent experiment history."""
    if not results:
        return "No experiments yet."
    recent = results[-n:]
    lines = []
    for r in recent:
        bpb = f"{r['post_quant_bpb']:.4f}" if r.get("post_quant_bpb") else "N/A"
        size = f"{r.get('size_bytes', 'N/A')}"
        gap = ""
        if r.get("pre_quant_bpb") and r.get("post_quant_bpb"):
            gap = f" (gap={r['post_quant_bpb'] - r['pre_quant_bpb']:.4f})"
        lines.append(
            f"  [{r['status']:>10}] {r['experiment_id'][:20]:20s} "
            f"bpb={bpb} size={size}{gap}"
        )
        # Include hypothesis if available
        exp = load_experiment(r["experiment_id"])
        if exp:
            lines.append(f"             → {exp.get('hypothesis', 'N/A')[:80]}")
    return "\n".join(lines)


def load_experiment(experiment_id: str) -> dict | None:
    """Load a specific experiment definition."""
    if not EXPERIMENTS_FILE.exists():
        return None
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            exp = json.loads(line)
            if exp.get("experiment_id") == experiment_id:
                return exp
    return None


def call_claude_agent(results: list[dict], api_key: str) -> dict:
    """Call Claude API to propose an experiment."""
    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = AGENT_SYSTEM_PROMPT.format(
        state_summary=format_state_summary(results),
        param_bounds=format_param_bounds(),
        available_patches=format_patches(),
        recent_history=format_history(results),
    )

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract JSON from response
    text = message.content[0].text
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try parsing the whole response as JSON
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError(f"Could not extract JSON from agent response:\n{text}")


# ---------------------
# Experiment runner
# ---------------------

def run_experiment(
    experiment: Experiment,
    proxy: bool = True,
    ngpu: int = 1,
    wallclock: int | None = None,
) -> ExperimentResult:
    """Run a single experiment and return structured results."""
    env = os.environ.copy()
    env.update(experiment.to_env_vars())
    env["RUN_ID"] = experiment.experiment_id

    cmd = ["bash", str(RUN_SCRIPT)]
    if proxy:
        cmd.append("--proxy")
    cmd.extend(["--ngpu", str(ngpu)])
    if wallclock is not None:
        cmd.extend(["--wallclock", str(wallclock)])

    print(f"\n{'='*60}")
    print(f"Running: {experiment.experiment_id}")
    print(f"  Hypothesis: {experiment.hypothesis}")
    print(f"  Overrides: {experiment.param_overrides}")
    print(f"  Patches: {experiment.code_patches}")
    print(f"  Stage: {'proxy' if proxy else 'full'}")
    print(f"{'='*60}\n")

    try:
        proc = subprocess.run(
            cmd,
            env=env,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min hard timeout
        )
        output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired:
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            pre_quant_bpb=None,
            post_quant_bpb=None,
            size_bytes=None,
            params=None,
            steps=None,
            status="timeout",
            stage="proxy" if proxy else "full",
            error_msg="Exceeded 30 min hard timeout",
        )
    except Exception as e:
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            pre_quant_bpb=None,
            post_quant_bpb=None,
            size_bytes=None,
            params=None,
            steps=None,
            status="crash",
            stage="proxy" if proxy else "full",
            error_msg=str(e),
        )

    if proc.returncode != 0:
        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            pre_quant_bpb=None,
            post_quant_bpb=None,
            size_bytes=None,
            params=None,
            steps=None,
            status="crash",
            stage="proxy" if proxy else "full",
            error_msg=output[-500:] if output else "Unknown error",
        )

    return parse_results(experiment.experiment_id, output, proxy)


def parse_results(experiment_id: str, output: str, proxy: bool) -> ExperimentResult:
    """Parse training output into structured results."""
    post_quant_bpb = None
    pre_quant_bpb = None
    size_bytes = None
    params = None
    steps = None

    # Post-quant BPB (the real metric)
    m = re.search(r"final_int8_zlib_roundtrip val_loss:\S+ val_bpb:(\S+)", output)
    if m:
        post_quant_bpb = float(m.group(1))

    # Pre-quant BPB (last validation line)
    for m in re.finditer(r"val_bpb:(\S+)", output):
        pre_quant_bpb = float(m.group(1))
    # The last match before final_int8 is the pre-quant one
    pre_quant_matches = re.findall(r"step:\d+/\d+ val_loss:\S+ val_bpb:(\S+)", output)
    if pre_quant_matches:
        pre_quant_bpb = float(pre_quant_matches[-1])

    # Size
    m = re.search(r"Total submission size int8\+zlib:\s*(\d+)\s*bytes", output)
    if m:
        size_bytes = int(m.group(1))

    # Params
    m = re.search(r"model_params:(\d+)", output)
    if m:
        params = int(m.group(1))

    # Steps
    step_matches = re.findall(r"step:(\d+)/", output)
    if step_matches:
        steps = int(step_matches[-1])

    status = "ok"
    if post_quant_bpb is None:
        status = "crash"
    elif size_bytes is not None and size_bytes >= 16_000_000:
        status = "over_budget"

    return ExperimentResult(
        experiment_id=experiment_id,
        pre_quant_bpb=pre_quant_bpb,
        post_quant_bpb=post_quant_bpb,
        size_bytes=size_bytes,
        params=params,
        steps=steps,
        status=status,
        stage="proxy" if proxy else "full",
    )


# ---------------------
# State management
# ---------------------

def load_results() -> list[dict]:
    """Load all experiment results."""
    if not RESULTS_FILE.exists():
        return []
    results = []
    with open(RESULTS_FILE) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def save_result(result: ExperimentResult):
    """Append a result to the results file."""
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result_to_dict(result)) + "\n")


def save_experiment(experiment: Experiment):
    """Append an experiment definition to the experiments file."""
    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(json.dumps(experiment_to_dict(experiment)) + "\n")


def get_best_bpb(results: list[dict], stage: str | None = None) -> float:
    """Get the best BPB from results, optionally filtered by stage."""
    valid = [
        r for r in results
        if r["status"] == "ok"
        and r.get("post_quant_bpb") is not None
        and (stage is None or r.get("stage") == stage)
    ]
    if not valid:
        return 1.2244  # baseline
    return min(r["post_quant_bpb"] for r in valid)


def get_best_config(results: list[dict]) -> dict:
    """Get the config that produced the best result."""
    valid = [r for r in results if r["status"] == "ok" and r.get("post_quant_bpb")]
    if not valid:
        return copy.deepcopy(BASELINE_CONFIG)
    best = min(valid, key=lambda r: r["post_quant_bpb"])
    exp_dict = load_experiment(best["experiment_id"])
    if exp_dict:
        exp = experiment_from_dict(exp_dict)
        return exp.effective_config()
    return copy.deepcopy(BASELINE_CONFIG)


# ---------------------
# Main loop
# ---------------------

def run_harness(
    agent: str = "claude",
    max_experiments: int = 100,
    ngpu: int = 1,
    proxy_wallclock: int = 25,
    full_wallclock: int = 600,
    resume: bool = False,
    from_file: str | None = None,
    full_ngpu: int = 8,
):
    """Main experiment loop."""
    results = load_results() if resume else []
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    if not resume and RESULTS_FILE.exists():
        RESULTS_FILE.rename(RESULTS_FILE.with_suffix(".jsonl.bak"))
    if not resume and EXPERIMENTS_FILE.exists():
        EXPERIMENTS_FILE.rename(EXPERIMENTS_FILE.with_suffix(".jsonl.bak"))

    print(f"Parameter Golf Autoresearch Harness")
    print(f"  Agent: {agent}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Proxy GPUs: {ngpu}, Full GPUs: {full_ngpu}")
    print(f"  Resume: {resume} ({len(results)} prior results)")
    print()

    # Queue for full-stage validation of promising proxy results
    full_validation_queue: list[Experiment] = []
    experiment_count = len(results)

    while experiment_count < max_experiments:
        # --- Check if we have proxy winners to validate at full scale ---
        if full_validation_queue and full_ngpu > 0:
            exp = full_validation_queue.pop(0)
            exp.stage = "full"
            exp.experiment_id = f"full_{exp.experiment_id}"
            print(f"\n*** PROMOTING TO FULL VALIDATION: {exp.experiment_id} ***\n")
            result = run_experiment(exp, proxy=False, ngpu=full_ngpu, wallclock=full_wallclock)
            save_result(result)
            results.append(result_to_dict(result))
            experiment_count += 1
            print_result_summary(result, results)
            continue

        # --- Propose next experiment ---
        try:
            if from_file:
                proposal = load_next_proposal(from_file, experiment_count)
                if proposal is None:
                    print("No more experiments in file.")
                    break
            elif agent == "claude":
                if not api_key:
                    print("ERROR: Set ANTHROPIC_API_KEY environment variable")
                    sys.exit(1)
                proposal = call_claude_agent(results, api_key)
            else:
                print(f"Unknown agent: {agent}")
                sys.exit(1)
        except Exception as e:
            print(f"Agent error: {e}")
            time.sleep(5)
            continue

        # --- Build experiment from proposal ---
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        experiment = Experiment(
            experiment_id=exp_id,
            parent_id=None,
            hypothesis=proposal.get("hypothesis", "No hypothesis provided"),
            param_overrides=proposal.get("param_overrides", {}),
            code_patches=proposal.get("code_patches", []),
            stage="proxy",
        )

        # --- Validate ---
        errors = validate_experiment(experiment)
        if errors:
            print(f"INVALID experiment proposal: {errors}")
            # Save as rejected so the agent learns
            rejected = ExperimentResult(
                experiment_id=exp_id,
                pre_quant_bpb=None,
                post_quant_bpb=None,
                size_bytes=None,
                params=None,
                steps=None,
                status="rejected",
                stage="proxy",
                error_msg="; ".join(errors),
            )
            save_result(rejected)
            results.append(result_to_dict(rejected))
            save_experiment(experiment)
            experiment_count += 1
            continue

        # --- Dedup check ---
        config_hash = experiment.config_hash()
        prior_hashes = set()
        if EXPERIMENTS_FILE.exists():
            with open(EXPERIMENTS_FILE) as f:
                for line in f:
                    prior = experiment_from_dict(json.loads(line))
                    prior_hashes.add(prior.config_hash())
        if config_hash in prior_hashes:
            print(f"DUPLICATE config (hash={config_hash}), skipping.")
            experiment_count += 1
            continue

        # --- Save and run ---
        save_experiment(experiment)
        result = run_experiment(experiment, proxy=True, ngpu=ngpu, wallclock=proxy_wallclock)
        save_result(result)
        results.append(result_to_dict(result))
        experiment_count += 1

        print_result_summary(result, results)

        # --- Promotion check ---
        current_best = get_best_bpb(results)
        if should_promote(result, current_best, "proxy"):
            print(f"\n*** EXPERIMENT QUALIFIES FOR FULL VALIDATION ***")
            full_validation_queue.append(copy.deepcopy(experiment))

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("AUTORESEARCH COMPLETE")
    print("=" * 60)
    print_final_summary(results)


def load_next_proposal(filepath: str, index: int) -> dict | None:
    """Load the Nth proposal from a JSONL file."""
    path = Path(filepath)
    if not path.exists():
        return None
    with open(path) as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    return None


def print_result_summary(result: ExperimentResult, all_results: list[dict]):
    """Print a summary of the latest result."""
    best_bpb = get_best_bpb(all_results)
    print(f"\n--- Result: {result.experiment_id} ---")
    print(f"  Status: {result.status}")
    if result.post_quant_bpb is not None:
        print(f"  Post-quant BPB: {result.post_quant_bpb:.4f}")
    if result.pre_quant_bpb is not None:
        print(f"  Pre-quant BPB: {result.pre_quant_bpb:.4f}")
    if result.quant_gap is not None:
        print(f"  Quant gap: {result.quant_gap:.4f}")
    if result.size_bytes is not None:
        print(f"  Size: {result.size_bytes:,} / 16,000,000 bytes")
    print(f"  Current best: {best_bpb:.4f}")
    print()


def print_final_summary(results: list[dict]):
    """Print final summary of all experiments."""
    total = len(results)
    ok = [r for r in results if r["status"] == "ok"]
    crashed = [r for r in results if r["status"] == "crash"]
    rejected = [r for r in results if r["status"] == "rejected"]
    over = [r for r in results if r["status"] == "over_budget"]

    print(f"Total experiments: {total}")
    print(f"  OK: {len(ok)}")
    print(f"  Crashed: {len(crashed)}")
    print(f"  Over budget: {len(over)}")
    print(f"  Rejected: {len(rejected)}")

    if ok:
        best = min(ok, key=lambda r: r["post_quant_bpb"])
        print(f"\nBest result:")
        print(f"  Experiment: {best['experiment_id']}")
        print(f"  Post-quant BPB: {best['post_quant_bpb']:.4f}")
        print(f"  Size: {best.get('size_bytes', 'N/A')}")

        exp = load_experiment(best["experiment_id"])
        if exp:
            print(f"  Hypothesis: {exp.get('hypothesis', 'N/A')}")
            print(f"  Overrides: {exp.get('param_overrides', {})}")
            print(f"  Patches: {exp.get('code_patches', [])}")


# ---------------------
# CLI
# ---------------------

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Autoresearch Harness")
    parser.add_argument("--agent", default="claude", choices=["claude"],
                        help="Agent to use for proposing experiments")
    parser.add_argument("--max-experiments", type=int, default=100,
                        help="Maximum number of experiments to run")
    parser.add_argument("--ngpu", type=int, default=0,
                        help="GPUs for proxy experiments (0=auto-detect)")
    parser.add_argument("--full-ngpu", type=int, default=0,
                        help="GPUs for full validation (0=skip full validation)")
    parser.add_argument("--proxy-wallclock", type=int, default=25,
                        help="Wallclock seconds for proxy experiments")
    parser.add_argument("--full-wallclock", type=int, default=600,
                        help="Wallclock seconds for full experiments")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    parser.add_argument("--from-file", type=str, default=None,
                        help="Load experiments from JSONL file instead of agent")
    args = parser.parse_args()

    # Auto-detect GPUs
    if args.ngpu == 0:
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
            )
            args.ngpu = len(result.stdout.strip().split("\n"))
        except (subprocess.CalledProcessError, FileNotFoundError):
            args.ngpu = 1

    run_harness(
        agent=args.agent,
        max_experiments=args.max_experiments,
        ngpu=args.ngpu,
        full_ngpu=args.full_ngpu,
        proxy_wallclock=args.proxy_wallclock,
        full_wallclock=args.full_wallclock,
        resume=args.resume,
        from_file=args.from_file,
    )


if __name__ == "__main__":
    main()
