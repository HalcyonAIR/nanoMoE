#!/usr/bin/env python3
"""
Severity Sweep: Quantify resilience across collapse strengths.

Answers: As collapse strength increases, how much extra stability does Chrono buy,
and what does it cost?

Metrics:
- Recovery time: Steps to return above Neff threshold after bias ends
- Min Neff: Lowest Neff during bias window
- Collapse area: Integrated Neff deficit during bias
- Dead expert events: Count and duration
- Governance energy: Mean pressure, mean scale, time at scale cap
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ResilienceMetrics:
    """Metrics for one experiment run."""
    condition: str
    bias_strength: float

    # Recovery
    recovery_time: int  # Steps after bias ends to reach threshold (or -1 if never)
    min_neff_during_bias: float  # Lowest Neff during bias window
    collapse_area: float  # Integrated (threshold - Neff) during bias

    # Dead experts
    dead_expert_steps: int  # Total step*layer instances with dead experts
    max_dead_count: int

    # Governance energy (controller ON only)
    mean_pressure: float
    mean_scale: float
    time_at_scale_cap: int  # Steps where scale was at max

    # Final state
    final_neff: float


def parse_metrics(jsonl_path: Path, is_controller_on: bool) -> Dict[int, Dict[int, dict]]:
    """Parse control_decisions.jsonl or snapshots.jsonl."""
    results = {}
    if not jsonl_path.exists():
        return results

    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            step = data['step']

            if 'layer_id' in data:  # control_decisions format
                layer_id = data['layer_id']
                if step not in results:
                    results[step] = {}
                results[step][layer_id] = {
                    'n_effective': data['observed']['n_effective'],
                    'dead_expert_count': int(data['observed']['dead_expert_count']),
                    'pressure': data['computed']['pressure'],
                    'lens_scale': data['actuator']['lens_scale'],
                }
            else:  # snapshots format
                if step not in results:
                    results[step] = {}
                for layer in data.get('layers', []):
                    layer_id = layer['layer_id']
                    results[step][layer_id] = {
                        'n_effective': layer['n_effective'],
                        'dead_expert_count': layer['dead_expert_count'],
                        'pressure': 0.0,
                        'lens_scale': 0.0,
                    }
    return results


def compute_resilience_metrics(
    metrics: Dict[int, Dict[int, dict]],
    condition: str,
    bias_strength: float,
    bias_start: int,
    bias_end: int,
    neff_threshold: float = 3.5,
    scale_cap: float = 0.5,
) -> ResilienceMetrics:
    """Compute resilience metrics from parsed data."""

    sorted_steps = sorted(metrics.keys())
    n_layers = max(max(m.keys()) for m in metrics.values()) + 1 if metrics else 0

    # Recovery time: first step after bias_end where ALL layers >= threshold
    recovery_time = -1
    for step in sorted_steps:
        if step <= bias_end:
            continue
        layer_metrics = metrics[step]
        all_recovered = all(
            m['n_effective'] >= neff_threshold
            for m in layer_metrics.values()
        )
        if all_recovered:
            recovery_time = step - bias_end
            break

    # Min Neff during bias
    min_neff = float('inf')
    for step in sorted_steps:
        if bias_start <= step <= bias_end:
            for m in metrics[step].values():
                min_neff = min(min_neff, m['n_effective'])
    if min_neff == float('inf'):
        min_neff = 0.0

    # Collapse area: integrated deficit during bias
    collapse_area = 0.0
    prev_step = None
    for step in sorted_steps:
        if bias_start <= step <= bias_end:
            for m in metrics[step].values():
                deficit = max(0, neff_threshold - m['n_effective'])
                if prev_step is not None:
                    dt = step - prev_step
                else:
                    dt = 1
                collapse_area += deficit * dt / n_layers  # Normalize by layers
            prev_step = step

    # Dead expert events
    dead_expert_steps = 0
    max_dead = 0
    for step, layer_metrics in metrics.items():
        for m in layer_metrics.values():
            if m['dead_expert_count'] > 0:
                dead_expert_steps += 1
                max_dead = max(max_dead, m['dead_expert_count'])

    # Governance energy
    pressures = []
    scales = []
    time_at_cap = 0
    for step, layer_metrics in metrics.items():
        for m in layer_metrics.values():
            pressures.append(m['pressure'])
            scales.append(m['lens_scale'])
            if m['lens_scale'] >= scale_cap * 0.99:  # Within 1% of cap
                time_at_cap += 1

    mean_pressure = np.mean(pressures) if pressures else 0.0
    mean_scale = np.mean(scales) if scales else 0.0

    # Final Neff
    if sorted_steps:
        final_step = sorted_steps[-1]
        final_neff = np.mean([m['n_effective'] for m in metrics[final_step].values()])
    else:
        final_neff = 0.0

    return ResilienceMetrics(
        condition=condition,
        bias_strength=bias_strength,
        recovery_time=recovery_time,
        min_neff_during_bias=min_neff,
        collapse_area=collapse_area,
        dead_expert_steps=dead_expert_steps,
        max_dead_count=max_dead,
        mean_pressure=mean_pressure,
        mean_scale=mean_scale,
        time_at_scale_cap=time_at_cap,
        final_neff=final_neff,
    )


def run_experiment(
    condition: str,
    seed: int,
    output_dir: Path,
    bias_strength: float,
    bias_start: int = 100,
    bias_end: int = 250,
    max_iters: int = 400,
) -> Optional[ResilienceMetrics]:
    """Run single experiment and return metrics."""

    run_dir = output_dir / f"run_{condition}_{bias_strength}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_content = f'''# Severity Sweep Config - {condition} bias={bias_strength}
out_dir = '{run_dir}'
eval_interval = 25
eval_iters = 20
log_interval = 10

always_save_checkpoint = False
wandb_log = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

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

use_chrono_controller = {condition == "ON"}
chrono_lens_rank = 4
chrono_neff_threshold_ratio = 0.85
chrono_top2_warning = 0.60

collapse_bias_expert_id = 0
collapse_bias_strength = {bias_strength}
collapse_bias_start_step = {bias_start}
collapse_bias_end_step = {bias_end}

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

    print(f"  Running: {condition} bias={bias_strength}...")

    result = subprocess.run(
        ['python', 'train.py', str(config_path)],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[:200]}")
        return None

    # Parse results
    telemetry_dir = run_dir / "telemetry"
    run_dirs = list(telemetry_dir.glob("run_*"))
    if not run_dirs:
        return None

    run_telemetry = run_dirs[0]

    if condition == "ON":
        metrics = parse_metrics(run_telemetry / "control_decisions.jsonl", True)
    else:
        metrics = parse_metrics(run_telemetry / "snapshots.jsonl", False)

    return compute_resilience_metrics(
        metrics, condition, bias_strength, bias_start, bias_end
    )


def plot_resilience_curves(
    on_results: List[ResilienceMetrics],
    off_results: List[ResilienceMetrics],
    output_path: Path,
):
    """Plot resilience curves comparing ON vs OFF."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    strengths_on = [r.bias_strength for r in on_results]
    strengths_off = [r.bias_strength for r in off_results]

    # Recovery time
    ax = axes[0, 0]
    recovery_on = [r.recovery_time if r.recovery_time >= 0 else 200 for r in on_results]
    recovery_off = [r.recovery_time if r.recovery_time >= 0 else 200 for r in off_results]
    ax.plot(strengths_on, recovery_on, 'b-o', label='Controller ON', linewidth=2)
    ax.plot(strengths_off, recovery_off, 'r--s', label='Controller OFF', linewidth=2)
    ax.set_xlabel('Bias Strength')
    ax.set_ylabel('Recovery Time (steps)')
    ax.set_title('Recovery Time After Bias')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Min Neff during bias
    ax = axes[0, 1]
    ax.plot(strengths_on, [r.min_neff_during_bias for r in on_results], 'b-o', label='Controller ON', linewidth=2)
    ax.plot(strengths_off, [r.min_neff_during_bias for r in off_results], 'r--s', label='Controller OFF', linewidth=2)
    ax.set_xlabel('Bias Strength')
    ax.set_ylabel('Min Neff')
    ax.set_title('Minimum Neff During Bias')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Collapse area
    ax = axes[0, 2]
    ax.plot(strengths_on, [r.collapse_area for r in on_results], 'b-o', label='Controller ON', linewidth=2)
    ax.plot(strengths_off, [r.collapse_area for r in off_results], 'r--s', label='Controller OFF', linewidth=2)
    ax.set_xlabel('Bias Strength')
    ax.set_ylabel('Collapse Area')
    ax.set_title('Integrated Neff Deficit')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dead expert steps
    ax = axes[1, 0]
    ax.plot(strengths_on, [r.dead_expert_steps for r in on_results], 'b-o', label='Controller ON', linewidth=2)
    ax.plot(strengths_off, [r.dead_expert_steps for r in off_results], 'r--s', label='Controller OFF', linewidth=2)
    ax.set_xlabel('Bias Strength')
    ax.set_ylabel('Dead Expert Events')
    ax.set_title('Dead Expert Step*Layer Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Governance energy (ON only)
    ax = axes[1, 1]
    ax.plot(strengths_on, [r.mean_pressure for r in on_results], 'g-o', label='Mean Pressure', linewidth=2)
    ax.plot(strengths_on, [r.mean_scale for r in on_results], 'm-^', label='Mean Scale', linewidth=2)
    ax.set_xlabel('Bias Strength')
    ax.set_ylabel('Value')
    ax.set_title('Governance Energy (Controller ON)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Resilience gain (ON - OFF)
    ax = axes[1, 2]
    min_neff_gain = [on.min_neff_during_bias - off.min_neff_during_bias
                     for on, off in zip(on_results, off_results)]
    ax.bar(strengths_on, min_neff_gain, color='green', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Bias Strength')
    ax.set_ylabel('Neff Gain (ON - OFF)')
    ax.set_title('Controller Benefit: Min Neff Improvement')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(on_results: List[ResilienceMetrics], off_results: List[ResilienceMetrics]):
    """Print summary table."""
    print("\n" + "="*80)
    print("SEVERITY SWEEP RESULTS")
    print("="*80)

    print(f"\n{'Bias':<8} {'Min Neff (ON/OFF)':<20} {'Recovery (ON/OFF)':<20} {'Dead Events':<15} {'Gov Energy':<12}")
    print("-"*80)

    for on, off in zip(on_results, off_results):
        rec_on = f"{on.recovery_time}" if on.recovery_time >= 0 else "Never"
        rec_off = f"{off.recovery_time}" if off.recovery_time >= 0 else "Never"
        print(f"{on.bias_strength:<8.1f} {on.min_neff_during_bias:.2f} / {off.min_neff_during_bias:.2f}      "
              f"{rec_on:>6} / {rec_off:<6}     {on.dead_expert_steps:>3} / {off.dead_expert_steps:<3}     "
              f"P={on.mean_pressure:.3f}")

    # Summary statistics
    print("\n" + "="*80)
    print("RESILIENCE SUMMARY")
    print("="*80)

    avg_neff_gain = np.mean([on.min_neff_during_bias - off.min_neff_during_bias
                            for on, off in zip(on_results, off_results)])
    avg_dead_reduction = np.mean([off.dead_expert_steps - on.dead_expert_steps
                                  for on, off in zip(on_results, off_results)])
    avg_pressure = np.mean([r.mean_pressure for r in on_results])

    print(f"Average Min Neff improvement: {avg_neff_gain:+.3f}")
    print(f"Average dead expert reduction: {avg_dead_reduction:+.1f} events")
    print(f"Average governance pressure: {avg_pressure:.3f}")

    if avg_neff_gain > 0:
        print("\n*** Controller provides measurable resilience benefit ***")
    else:
        print("\n*** No clear resilience benefit detected ***")


def main():
    """Run severity sweep."""

    # Parameters
    seed = 1337
    bias_strengths = [5.0, 10.0, 15.0, 20.0]
    bias_start = 100
    bias_end = 250
    max_iters = 400

    output_dir = Path("out-severity-sweep")
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("SEVERITY SWEEP: Quantifying Resilience")
    print("="*60)
    print(f"Bias strengths: {bias_strengths}")
    print(f"Bias window: steps {bias_start}-{bias_end}")
    print()

    on_results = []
    off_results = []

    for strength in bias_strengths:
        print(f"\nBias strength = {strength}")

        # Run OFF first
        off_result = run_experiment(
            condition="OFF",
            seed=seed,
            output_dir=output_dir,
            bias_strength=strength,
            bias_start=bias_start,
            bias_end=bias_end,
            max_iters=max_iters,
        )
        if off_result:
            off_results.append(off_result)

        # Run ON
        on_result = run_experiment(
            condition="ON",
            seed=seed,
            output_dir=output_dir,
            bias_strength=strength,
            bias_start=bias_start,
            bias_end=bias_end,
            max_iters=max_iters,
        )
        if on_result:
            on_results.append(on_result)

    if on_results and off_results:
        # Plot
        plot_resilience_curves(on_results, off_results, output_dir / "resilience_curves.png")

        # Print summary
        print_summary(on_results, off_results)

        # Save raw data
        with open(output_dir / "results.json", 'w') as f:
            json.dump({
                'on': [asdict(r) for r in on_results],
                'off': [asdict(r) for r in off_results],
            }, f, indent=2)

        print(f"\nResults saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
