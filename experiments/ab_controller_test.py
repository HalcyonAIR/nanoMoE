#!/usr/bin/env python3
"""
A/B Controller Test: Prove causal benefit of ChronoMoE Phase 2 controller.

Same seed, same forced pathology schedule, only difference = controller ON vs OFF.

Success criteria (defined in advance):
- Controller ON recovers Neff faster after bias ends, AND/OR
- Controller ON prevents dead experts that occur with OFF

Metrics logged per eval step:
- Neff per layer
- Top2 share per layer
- Dead expert count per layer
- Pressure (controller ON only)
- Lens scale (controller ON only)
- Recovery time after bias ends
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvalMetrics:
    """Metrics captured at each eval step."""
    step: int
    layer_id: int
    n_effective: float
    top2_share: float
    dead_expert_count: int
    pressure: float = 0.0  # Only for controller ON
    lens_scale: float = 0.0  # Only for controller ON


@dataclass
class ExperimentResult:
    """Results from one experiment run."""
    condition: str  # "ON" or "OFF"
    seed: int
    metrics: List[EvalMetrics] = field(default_factory=list)
    recovery_step: Optional[int] = None  # Step when Neff recovered after bias
    had_dead_experts: bool = False
    max_dead_count: int = 0


def parse_control_decisions(jsonl_path: Path) -> Dict[int, Dict[int, dict]]:
    """Parse control_decisions.jsonl into {step: {layer_id: metrics}}."""
    results = {}
    if not jsonl_path.exists():
        return results

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            step = data['step']
            layer_id = data['layer_id']
            if step not in results:
                results[step] = {}
            results[step][layer_id] = {
                'n_effective': data['observed']['n_effective'],
                'top2_share': data['observed']['top2_share'],
                'dead_expert_count': int(data['observed']['dead_expert_count']),
                'pressure': data['computed']['pressure'],
                'lens_scale': data['actuator']['lens_scale'],
            }
    return results


def parse_snapshots(jsonl_path: Path) -> Dict[int, Dict[int, dict]]:
    """Parse snapshots.jsonl for controller OFF runs."""
    results = {}
    if not jsonl_path.exists():
        return results

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            step = data['step']
            if step not in results:
                results[step] = {}
            for layer in data.get('layers', []):
                layer_id = layer['layer_id']
                results[step][layer_id] = {
                    'n_effective': layer['n_effective'],
                    'top2_share': layer['top2_share'],
                    'dead_expert_count': layer['dead_expert_count'],
                    'pressure': 0.0,
                    'lens_scale': 0.0,
                }
    return results


def compute_recovery_step(
    metrics: Dict[int, Dict[int, dict]],
    bias_end_step: int,
    recovery_threshold: float = 3.5,  # Neff threshold for "recovered"
) -> Optional[int]:
    """Find first step after bias_end where all layers have Neff >= threshold."""
    sorted_steps = sorted(s for s in metrics.keys() if s > bias_end_step)
    for step in sorted_steps:
        layer_metrics = metrics[step]
        all_recovered = all(
            m['n_effective'] >= recovery_threshold
            for m in layer_metrics.values()
        )
        if all_recovered:
            return step
    return None


def run_experiment(
    condition: str,
    seed: int,
    output_dir: Path,
    bias_strength: float = 15.0,
    bias_start: int = 100,
    bias_end: int = 250,
    max_iters: int = 500,
) -> ExperimentResult:
    """Run single experiment with controller ON or OFF."""

    run_dir = output_dir / f"run_{condition}_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build config file
    config_content = f'''# A/B Test Config - Controller {condition}
out_dir = '{run_dir}'
eval_interval = 25  # Frequent evals for fine-grained recovery tracking
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

# Fixed seed for reproducibility
# Note: seed is set via torch.manual_seed in train.py

# Small model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# MoE settings
n_exp = 4
top_k = 2
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = 1.25
eval_capacity = 2.0
stride = 2
use_switch_tfm_init = True
router_use_full_prec = True

# ChronoMoE Phase 2 - THE ONLY DIFFERENCE
use_chrono_controller = {condition == "ON"}
chrono_lens_rank = 4
chrono_neff_threshold_ratio = 0.85
chrono_top2_warning = 0.60

# Forced pathology schedule (same for both)
collapse_bias_expert_id = 0
collapse_bias_strength = {bias_strength}
collapse_bias_start_step = {bias_start}
collapse_bias_end_step = {bias_end}

# Training
learning_rate = 1e-3
max_iters = {max_iters}
lr_decay_iters = {max_iters}
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 50

device = 'mps'
compile = False
'''

    config_path = run_dir / "config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Run training
    print(f"\n{'='*60}")
    print(f"Running: Controller {condition}, Seed {seed}")
    print(f"Bias: strength={bias_strength}, steps {bias_start}-{bias_end}")
    print(f"{'='*60}\n")

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'

    result = subprocess.run(
        ['python', 'train.py', str(config_path)],
        cwd=Path(__file__).parent.parent,
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        return ExperimentResult(condition=condition, seed=seed)

    # Parse results
    telemetry_dir = run_dir / "telemetry"
    run_dirs = list(telemetry_dir.glob("run_*"))
    if not run_dirs:
        print(f"No telemetry found in {telemetry_dir}")
        return ExperimentResult(condition=condition, seed=seed)

    run_telemetry = run_dirs[0]

    # Parse metrics based on condition
    if condition == "ON":
        metrics = parse_control_decisions(run_telemetry / "control_decisions.jsonl")
    else:
        metrics = parse_snapshots(run_telemetry / "snapshots.jsonl")

    # Build result
    exp_result = ExperimentResult(condition=condition, seed=seed)

    for step in sorted(metrics.keys()):
        for layer_id, m in metrics[step].items():
            exp_result.metrics.append(EvalMetrics(
                step=step,
                layer_id=layer_id,
                n_effective=m['n_effective'],
                top2_share=m['top2_share'],
                dead_expert_count=m['dead_expert_count'],
                pressure=m['pressure'],
                lens_scale=m['lens_scale'],
            ))
            if m['dead_expert_count'] > 0:
                exp_result.had_dead_experts = True
                exp_result.max_dead_count = max(
                    exp_result.max_dead_count,
                    m['dead_expert_count']
                )

    # Compute recovery
    exp_result.recovery_step = compute_recovery_step(metrics, bias_end)

    return exp_result


def plot_comparison(
    on_result: ExperimentResult,
    off_result: ExperimentResult,
    output_path: Path,
    bias_start: int,
    bias_end: int,
):
    """Plot A/B comparison."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Extract data by layer
    def get_layer_data(result: ExperimentResult, layer_id: int):
        return [(m.step, m) for m in result.metrics if m.layer_id == layer_id]

    for layer_id in [0, 1]:
        on_data = get_layer_data(on_result, layer_id)
        off_data = get_layer_data(off_result, layer_id)

        on_steps = [d[0] for d in on_data]
        off_steps = [d[0] for d in off_data]

        # Neff
        ax = axes[0, layer_id]
        ax.plot(on_steps, [d[1].n_effective for d in on_data], 'b-', label='Controller ON', linewidth=2)
        ax.plot(off_steps, [d[1].n_effective for d in off_data], 'r--', label='Controller OFF', linewidth=2)
        ax.axvspan(bias_start, bias_end, alpha=0.2, color='red', label='Bias Window')
        ax.axhline(y=3.5, color='green', linestyle=':', alpha=0.5, label='Recovery Threshold')
        ax.set_ylabel('N_effective')
        ax.set_title(f'Layer {layer_id}: N_effective')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Top2 Share
        ax = axes[1, layer_id]
        ax.plot(on_steps, [d[1].top2_share for d in on_data], 'b-', label='Controller ON', linewidth=2)
        ax.plot(off_steps, [d[1].top2_share for d in off_data], 'r--', label='Controller OFF', linewidth=2)
        ax.axvspan(bias_start, bias_end, alpha=0.2, color='red')
        ax.set_ylabel('Top2 Share')
        ax.set_title(f'Layer {layer_id}: Top2 Share')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Dead Experts / Pressure
        ax = axes[2, layer_id]
        ax.plot(on_steps, [d[1].dead_expert_count for d in on_data], 'b-', label='Dead (ON)', linewidth=2)
        ax.plot(off_steps, [d[1].dead_expert_count for d in off_data], 'r--', label='Dead (OFF)', linewidth=2)
        if on_data and on_data[0][1].pressure > 0:
            ax2 = ax.twinx()
            ax2.plot(on_steps, [d[1].pressure for d in on_data], 'g:', label='Pressure', linewidth=1.5)
            ax2.set_ylabel('Pressure', color='green')
        ax.axvspan(bias_start, bias_end, alpha=0.2, color='red')
        ax.set_ylabel('Dead Expert Count')
        ax.set_xlabel('Step')
        ax.set_title(f'Layer {layer_id}: Dead Experts & Pressure')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def print_summary(on_result: ExperimentResult, off_result: ExperimentResult, bias_end: int):
    """Print summary comparison."""
    print("\n" + "="*60)
    print("A/B TEST RESULTS")
    print("="*60)

    print(f"\n{'Metric':<30} {'Controller ON':<20} {'Controller OFF':<20}")
    print("-"*70)

    # Recovery
    on_recovery = on_result.recovery_step
    off_recovery = off_result.recovery_step
    on_str = f"Step {on_recovery}" if on_recovery else "Never"
    off_str = f"Step {off_recovery}" if off_recovery else "Never"
    print(f"{'Recovery Step (Neff >= 3.5)':<30} {on_str:<20} {off_str:<20}")

    if on_recovery and off_recovery:
        on_time = on_recovery - bias_end
        off_time = off_recovery - bias_end
        print(f"{'Recovery Time (steps)':<30} {on_time:<20} {off_time:<20}")

    # Dead experts
    print(f"{'Had Dead Experts':<30} {str(on_result.had_dead_experts):<20} {str(off_result.had_dead_experts):<20}")
    print(f"{'Max Dead Count':<30} {on_result.max_dead_count:<20} {off_result.max_dead_count:<20}")

    # Final Neff (last step, average across layers)
    def get_final_neff(result):
        final_step = max(m.step for m in result.metrics) if result.metrics else 0
        final_metrics = [m for m in result.metrics if m.step == final_step]
        return np.mean([m.n_effective for m in final_metrics]) if final_metrics else 0

    on_final = get_final_neff(on_result)
    off_final = get_final_neff(off_result)
    print(f"{'Final Avg Neff':<30} {on_final:<20.3f} {off_final:<20.3f}")

    # Success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*60)

    success = False
    reasons = []

    # Criterion 1: Faster recovery
    if on_recovery and off_recovery and on_recovery < off_recovery:
        success = True
        reasons.append(f"ON recovered {off_recovery - on_recovery} steps faster")
    elif on_recovery and not off_recovery:
        success = True
        reasons.append("ON recovered, OFF never did")

    # Criterion 2: Prevented dead experts
    if off_result.had_dead_experts and not on_result.had_dead_experts:
        success = True
        reasons.append("ON prevented dead experts that occurred with OFF")
    elif off_result.max_dead_count > on_result.max_dead_count:
        success = True
        reasons.append(f"ON had fewer dead experts ({on_result.max_dead_count} vs {off_result.max_dead_count})")

    # Criterion 3: Better final state
    if on_final > off_final + 0.1:
        success = True
        reasons.append(f"ON has better final Neff ({on_final:.2f} vs {off_final:.2f})")

    if success:
        print("\n*** SUCCESS: Controller ON beats OFF ***")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("\n*** FAIL: Controller ON does not measurably beat OFF ***")
        print("  Need to tune controller before proceeding to distillation")

    return success


def main():
    """Run A/B controller test."""
    # Fixed parameters
    seed = 1337
    bias_strength = 15.0
    bias_start = 100
    bias_end = 250
    max_iters = 500  # Run longer for recovery observation

    output_dir = Path("out-ab-test")
    output_dir.mkdir(exist_ok=True)

    # Run both conditions
    off_result = run_experiment(
        condition="OFF",
        seed=seed,
        output_dir=output_dir,
        bias_strength=bias_strength,
        bias_start=bias_start,
        bias_end=bias_end,
        max_iters=max_iters,
    )

    on_result = run_experiment(
        condition="ON",
        seed=seed,
        output_dir=output_dir,
        bias_strength=bias_strength,
        bias_start=bias_start,
        bias_end=bias_end,
        max_iters=max_iters,
    )

    # Plot comparison
    plot_comparison(
        on_result=on_result,
        off_result=off_result,
        output_path=output_dir / "ab_comparison.png",
        bias_start=bias_start,
        bias_end=bias_end,
    )

    # Print summary and evaluate success
    success = print_summary(on_result, off_result, bias_end)

    # Save raw results
    with open(output_dir / "results.json", 'w') as f:
        json.dump({
            'on': {
                'condition': on_result.condition,
                'seed': on_result.seed,
                'recovery_step': on_result.recovery_step,
                'had_dead_experts': on_result.had_dead_experts,
                'max_dead_count': on_result.max_dead_count,
                'metrics': [asdict(m) for m in on_result.metrics],
            },
            'off': {
                'condition': off_result.condition,
                'seed': off_result.seed,
                'recovery_step': off_result.recovery_step,
                'had_dead_experts': off_result.had_dead_experts,
                'max_dead_count': off_result.max_dead_count,
                'metrics': [asdict(m) for m in off_result.metrics],
            },
            'success': success,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
