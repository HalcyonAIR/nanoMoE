#!/usr/bin/env python3
"""
Distillation Test: Does the controller help during knowledge distillation?

Hypothesis: KD creates different pressure on the router than normal training.
The controller should help if distillation lives in the "mild perturbation" regime,
and hurt if it lives in the "geometry-sensitive" intermediate regime.

This experiment:
1. Trains a teacher model to convergence (no controller)
2. Trains student with KD loss, comparing controller ON vs OFF
3. Measures topology metrics throughout
4. Determines which regime distillation occupies

Key metrics:
- Final loss (task performance)
- Neff trajectory during distillation
- Dead expert events
- Abstention frequency (new: explicit abstain mode)
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DistillationMetrics:
    """Metrics for one distillation run."""
    condition: str  # "ON" or "OFF"

    # Task performance
    final_train_loss: float
    final_val_loss: float

    # Topology health
    min_neff: float
    mean_neff: float
    final_neff: float
    dead_expert_events: int

    # Controller behavior (ON only)
    mean_pressure: float
    mean_scale: float
    abstain_count: int  # NEW: explicit abstention events
    abstain_ratio: float  # Fraction of steps where controller abstained


def parse_telemetry(telemetry_dir: Path, is_controller_on: bool) -> Dict:
    """Parse telemetry files for metrics."""
    results = {
        'neffs': [],
        'pressures': [],
        'scales': [],
        'dead_events': 0,
        'abstain_count': 0,
        'total_decisions': 0,
    }

    if is_controller_on:
        decisions_file = telemetry_dir / 'control_decisions.jsonl'
        if decisions_file.exists():
            with open(decisions_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    results['neffs'].append(data['observed']['n_effective'])
                    results['pressures'].append(data['computed']['pressure'])
                    results['scales'].append(data['actuator']['lens_scale'])
                    results['total_decisions'] += 1

                    if data['observed']['dead_expert_count'] > 0:
                        results['dead_events'] += 1

                    # Count explicit abstentions
                    if data['computed'].get('abstain', False):
                        results['abstain_count'] += 1
    else:
        snapshots_file = telemetry_dir / 'snapshots.jsonl'
        if snapshots_file.exists():
            with open(snapshots_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    for layer in data.get('layers', []):
                        results['neffs'].append(layer['n_effective'])
                        if layer['dead_expert_count'] > 0:
                            results['dead_events'] += 1
                        results['total_decisions'] += 1

    return results


def parse_training_log(log_file: Path) -> Dict[str, float]:
    """Parse training log for final losses."""
    losses = {'train': None, 'val': None}

    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                # Look for final loss values
                if 'train loss' in line.lower():
                    try:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if 'loss' in p.lower() and i+1 < len(parts):
                                val = float(parts[i+1].rstrip(','))
                                if 'train' in line.lower():
                                    losses['train'] = val
                                elif 'val' in line.lower():
                                    losses['val'] = val
                    except (ValueError, IndexError):
                        pass

    return losses


def run_training(
    config_content: str,
    run_dir: Path,
    description: str
) -> bool:
    """Run training with given config."""
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"  Running: {description}...")

    result = subprocess.run(
        ['python', 'train.py', str(config_path)],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[:300]}")
        return False

    return True


def run_distillation_experiment(
    output_dir: Path,
    teacher_iters: int = 300,
    student_iters: int = 200,
    kd_temperature: float = 2.0,
    kd_alpha: float = 0.5,
) -> Dict[str, DistillationMetrics]:
    """
    Run full distillation experiment.

    Phase 1: Train teacher (no controller)
    Phase 2: Train student with KD, compare ON vs OFF
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Common config base
    base_config = f'''# Distillation Test Config
eval_interval = 25
eval_iters = 20
log_interval = 10

always_save_checkpoint = True
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

chrono_lens_rank = 4
chrono_neff_threshold_ratio = 0.85
chrono_top2_warning = 0.60

learning_rate = 1e-3
beta2 = 0.99
warmup_iters = 50

device = 'mps'
compile = False
'''

    # =========================================================================
    # Phase 1: Train teacher model
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 1: Training Teacher Model")
    print("="*60)

    teacher_dir = output_dir / "teacher"
    teacher_config = base_config + f'''
out_dir = '{teacher_dir}'
use_chrono_controller = False
max_iters = {teacher_iters}
lr_decay_iters = {teacher_iters}
min_lr = 1e-4
'''

    if not run_training(teacher_config, teacher_dir, "Teacher (no controller)"):
        print("Teacher training failed!")
        return results

    # Verify teacher checkpoint exists
    teacher_ckpt = teacher_dir / "ckpt.pt"
    if not teacher_ckpt.exists():
        print(f"Teacher checkpoint not found at {teacher_ckpt}")
        return results

    print(f"  Teacher checkpoint saved: {teacher_ckpt}")

    # =========================================================================
    # Phase 2: Train students with KD
    # =========================================================================
    print("\n" + "="*60)
    print("PHASE 2: Knowledge Distillation (Student Training)")
    print("="*60)
    print(f"  KD Temperature: {kd_temperature}")
    print(f"  KD Alpha: {kd_alpha}")
    print(f"  Teacher: {teacher_ckpt}")

    for condition in ["OFF", "ON"]:
        print(f"\n--- Student with Controller {condition} ---")

        student_dir = output_dir / f"student_{condition}"
        student_config = base_config + f'''
out_dir = '{student_dir}'
use_chrono_controller = {condition == "ON"}
max_iters = {student_iters}
lr_decay_iters = {student_iters}
min_lr = 1e-4

# Knowledge Distillation settings
use_kd_loss = True
kd_teacher_path = '{teacher_ckpt}'
kd_temperature = {kd_temperature}
kd_alpha = {kd_alpha}
'''

        if not run_training(student_config, student_dir, f"Student (controller {condition})"):
            print(f"  Student {condition} training failed!")
            continue

        # Parse results
        telemetry_dirs = list((student_dir / "telemetry").glob("run_*"))
        if not telemetry_dirs:
            print(f"  No telemetry found for {condition}")
            continue

        telemetry = parse_telemetry(telemetry_dirs[0], condition == "ON")

        # Compute metrics
        neffs = telemetry['neffs'] if telemetry['neffs'] else [0]
        pressures = telemetry['pressures'] if telemetry['pressures'] else [0]
        scales = telemetry['scales'] if telemetry['scales'] else [0]

        metrics = DistillationMetrics(
            condition=condition,
            final_train_loss=0.0,  # TODO: parse from log
            final_val_loss=0.0,
            min_neff=min(neffs) if neffs else 0,
            mean_neff=np.mean(neffs) if neffs else 0,
            final_neff=neffs[-1] if neffs else 0,
            dead_expert_events=telemetry['dead_events'],
            mean_pressure=np.mean(pressures) if pressures else 0,
            mean_scale=np.mean(scales) if scales else 0,
            abstain_count=telemetry['abstain_count'],
            abstain_ratio=telemetry['abstain_count'] / max(1, telemetry['total_decisions']),
        )

        results[condition] = metrics

        print(f"  Results for {condition}:")
        print(f"    Min Neff: {metrics.min_neff:.2f}")
        print(f"    Mean Neff: {metrics.mean_neff:.2f}")
        print(f"    Dead events: {metrics.dead_expert_events}")
        if condition == "ON":
            print(f"    Mean pressure: {metrics.mean_pressure:.3f}")
            print(f"    Mean scale: {metrics.mean_scale:.3f}")
            print(f"    Abstain ratio: {metrics.abstain_ratio:.1%}")

    return results


def print_summary(results: Dict[str, DistillationMetrics]):
    """Print comparison summary."""
    if "ON" not in results or "OFF" not in results:
        print("\nIncomplete results - cannot compare")
        return

    on = results["ON"]
    off = results["OFF"]

    print("\n" + "="*70)
    print("DISTILLATION EXPERIMENT RESULTS")
    print("="*70)

    print(f"\n{'Metric':<25} {'Controller ON':<15} {'Controller OFF':<15} {'Delta':<10}")
    print("-"*70)

    metrics = [
        ("Min Neff", on.min_neff, off.min_neff),
        ("Mean Neff", on.mean_neff, off.mean_neff),
        ("Final Neff", on.final_neff, off.final_neff),
        ("Dead Expert Events", on.dead_expert_events, off.dead_expert_events),
    ]

    for name, on_val, off_val in metrics:
        delta = on_val - off_val
        print(f"{name:<25} {on_val:<15.2f} {off_val:<15.2f} {delta:+.2f}")

    print(f"\nController-specific metrics:")
    print(f"  Mean pressure: {on.mean_pressure:.3f}")
    print(f"  Mean scale: {on.mean_scale:.3f}")
    print(f"  Abstain ratio: {on.abstain_ratio:.1%} ({on.abstain_count} events)")

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS: Which regime does distillation occupy?")
    print("="*70)

    neff_benefit = on.min_neff - off.min_neff
    dead_benefit = off.dead_expert_events - on.dead_expert_events

    if neff_benefit > 0 and dead_benefit >= 0:
        regime = "MILD PERTURBATION"
        verdict = "Controller helps - safe for distillation"
    elif neff_benefit < -0.3 and dead_benefit < 0:
        regime = "GEOMETRY-SENSITIVE INTERMEDIATE"
        verdict = "Controller hurts - need mode selection"
    elif on.abstain_ratio > 0.3:
        regime = "HIGH SEVERITY (ABSTAINING)"
        verdict = "Controller correctly abstains - damage limited"
    else:
        regime = "UNCLEAR"
        verdict = "Need more data or tuning"

    print(f"\nRegime: {regime}")
    print(f"Verdict: {verdict}")

    if on.abstain_ratio > 0.1:
        print(f"\nNote: Controller abstained {on.abstain_ratio:.0%} of the time")
        print("This indicates the harm guard is actively protecting against bad steering")


def main():
    """Run distillation experiment."""
    output_dir = Path("out-distillation")

    print("="*60)
    print("DISTILLATION TEST: Controller Under KD Pressure")
    print("="*60)
    print("\nThis experiment tests whether the controller helps or hurts")
    print("during knowledge distillation, which creates different router")
    print("pressure than standard training.")
    print()

    results = run_distillation_experiment(
        output_dir=output_dir,
        teacher_iters=300,
        student_iters=200,
        kd_temperature=2.0,
        kd_alpha=0.5,
    )

    if results:
        print_summary(results)

        # Save results
        with open(output_dir / "results.json", 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)

        print(f"\nResults saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
