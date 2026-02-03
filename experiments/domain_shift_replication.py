#!/usr/bin/env python3
"""
Replication: Domain-Shift + Capacity Stress (Variant)

This replicates the stressed domain-shift experiment with a DIFFERENT
stress configuration to verify the protective envelope generalizes.

Original: top_k=1, train_capacity=1.25
This run: top_k=1, train_capacity=1.0 (tighter capacity)

Same controller, same logging, different stressor intensity.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ReplicationMetrics:
    """Metrics for replication run."""
    condition: str
    stressor: str  # Description of stress variant

    # Primary metrics
    dead_expert_events: int
    recovery_steps: int
    final_neff: float

    # Secondary
    min_neff: float
    mean_neff: float

    # Controller behavior
    nonzero_pressure_ratio: float
    mean_pressure: float
    max_pressure: float
    mean_scale: float
    abstain_ratio: float


def parse_telemetry(telemetry_dir: Path, is_controller_on: bool, recovery_threshold: float = 2.4) -> Dict:
    """Parse telemetry files."""
    results = {
        'neffs': [],
        'pressures': [],
        'scales': [],
        'dead_events': 0,
        'abstain_count': 0,
        'nonzero_pressure_count': 0,
        'total_decisions': 0,
        'recovery_step': -1,
    }

    saw_collapse = False

    if is_controller_on:
        decisions_file = telemetry_dir / 'control_decisions.jsonl'
        if decisions_file.exists():
            with open(decisions_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    step = data.get('step', 0)
                    neff = data['observed']['n_effective']
                    pressure = data['computed']['pressure']

                    results['neffs'].append(neff)
                    results['pressures'].append(pressure)
                    results['scales'].append(data['actuator']['lens_scale'])
                    results['total_decisions'] += 1

                    if pressure > 0.01:
                        results['nonzero_pressure_count'] += 1
                    if data['observed']['dead_expert_count'] > 0:
                        results['dead_events'] += 1
                    if data['computed'].get('abstain', False):
                        results['abstain_count'] += 1

                    if neff < recovery_threshold:
                        saw_collapse = True
                    elif saw_collapse and neff >= recovery_threshold and results['recovery_step'] < 0:
                        results['recovery_step'] = step
    else:
        snapshots_file = telemetry_dir / 'snapshots.jsonl'
        if snapshots_file.exists():
            with open(snapshots_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    step = data.get('step', 0)
                    for layer in data.get('layers', []):
                        neff = layer['n_effective']
                        results['neffs'].append(neff)
                        results['total_decisions'] += 1
                        if layer['dead_expert_count'] > 0:
                            results['dead_events'] += 1
                        if layer['layer_id'] == 0:
                            if neff < recovery_threshold:
                                saw_collapse = True
                            elif saw_collapse and neff >= recovery_threshold and results['recovery_step'] < 0:
                                results['recovery_step'] = step

    return results


def run_training(config_content: str, run_dir: Path, description: str) -> bool:
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
        print(f"    FAILED: {result.stderr[:500]}")
        return False
    return True


def run_replication(
    output_dir: Path,
    pretrain_iters: int = 500,
    finetune_iters: int = 300,
    finetune_top_k: int = 1,
    finetune_capacity: float = 1.0,  # Tighter than original 1.25
) -> Dict[str, ReplicationMetrics]:
    """Run replication with different stress configuration."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    stressor_desc = f"top_k={finetune_top_k}, capacity={finetune_capacity}"

    # Pre-train config (standard)
    pretrain_config = f'''# Pre-training Config
eval_interval = 25
eval_iters = 20
log_interval = 10
always_save_checkpoint = True
wandb_log = False
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

    # Fine-tune config (stressed with tighter capacity)
    finetune_base = f'''# Fine-tuning Config (STRESSED: {stressor_desc})
eval_interval = 25
eval_iters = 20
log_interval = 10
always_save_checkpoint = True
wandb_log = False
gradient_accumulation_steps = 1
batch_size = 32
block_size = 128
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1
n_exp = 4
top_k = {finetune_top_k}
use_aux_loss = True
aux_loss_weight = 0.01
use_router_z_loss = True
router_z_loss_weight = 0.001
use_noisy_top_k = False
train_capacity = {finetune_capacity}
eval_capacity = 2.0
stride = 2
use_switch_tfm_init = True
router_use_full_prec = True
chrono_lens_rank = 4
chrono_neff_threshold_ratio = 0.85
chrono_top2_warning = 0.60
learning_rate = 3e-4
beta2 = 0.99
warmup_iters = 50
device = 'mps'
compile = False
'''

    # Phase 1: Pre-train
    print("\n" + "="*70)
    print("REPLICATION: Pre-training on Shakespeare")
    print("="*70)

    pretrain_dir = output_dir / "pretrain"
    full_pretrain = pretrain_config + f'''
out_dir = '{pretrain_dir}'
dataset = 'shakespeare_char'
use_chrono_controller = False
max_iters = {pretrain_iters}
lr_decay_iters = {pretrain_iters}
min_lr = 1e-4
'''

    if not run_training(full_pretrain, pretrain_dir, "Pre-training"):
        return results

    pretrain_ckpt = pretrain_dir / "ckpt.pt"
    if not pretrain_ckpt.exists():
        print("Pre-train checkpoint not found")
        return results

    # Phase 2: Fine-tune with stress variant
    print("\n" + "="*70)
    print(f"REPLICATION: Fine-tuning with stress ({stressor_desc})")
    print("="*70)

    for condition in ["OFF", "ON"]:
        print(f"\n--- Controller {condition} ---")

        finetune_dir = output_dir / f"finetune_{condition}"
        finetune_config = finetune_base + f'''
out_dir = '{finetune_dir}'
dataset = 'tinystories'
use_chrono_controller = {condition == "ON"}
max_iters = {finetune_iters}
lr_decay_iters = {finetune_iters}
min_lr = 1e-4
init_from = 'finetune'
ckpt_path = '{pretrain_ckpt}'
'''

        if not run_training(finetune_config, finetune_dir, f"Fine-tuning ({condition})"):
            continue

        telemetry_dirs = list((finetune_dir / "telemetry").glob("run_*"))
        if not telemetry_dirs:
            continue

        telemetry = parse_telemetry(telemetry_dirs[0], condition == "ON")
        neffs = telemetry['neffs'] if telemetry['neffs'] else [0]
        pressures = telemetry['pressures'] if telemetry['pressures'] else [0]
        scales = telemetry['scales'] if telemetry['scales'] else [0]

        recovery = telemetry['recovery_step'] if telemetry['recovery_step'] >= 0 else finetune_iters

        metrics = ReplicationMetrics(
            condition=condition,
            stressor=stressor_desc,
            dead_expert_events=telemetry['dead_events'],
            recovery_steps=recovery,
            final_neff=neffs[-1] if neffs else 0,
            min_neff=min(neffs) if neffs else 0,
            mean_neff=float(np.mean(neffs)) if neffs else 0,
            nonzero_pressure_ratio=telemetry['nonzero_pressure_count'] / max(1, telemetry['total_decisions']),
            mean_pressure=float(np.mean(pressures)) if pressures else 0,
            max_pressure=float(max(pressures)) if pressures else 0,
            mean_scale=float(np.mean(scales)) if scales else 0,
            abstain_ratio=telemetry['abstain_count'] / max(1, telemetry['total_decisions']),
        )

        results[condition] = metrics
        print(f"  Dead: {metrics.dead_expert_events}, Final Neff: {metrics.final_neff:.2f}, "
              f"Nonzero pressure: {metrics.nonzero_pressure_ratio:.1%}")

    return results


def print_comparison(original: Dict, replication: Dict):
    """Print side-by-side comparison of original and replication."""
    print("\n" + "="*70)
    print("REPLICATION COMPARISON")
    print("="*70)

    print(f"\n{'Metric':<20} {'Original ON':<12} {'Original OFF':<12} {'Replic ON':<12} {'Replic OFF':<12}")
    print("-"*70)

    if "ON" in original and "OFF" in original and "ON" in replication and "OFF" in replication:
        o_on, o_off = original["ON"], original["OFF"]
        r_on, r_off = replication["ON"], replication["OFF"]

        rows = [
            ("Dead Events", o_on.dead_expert_events, o_off.dead_expert_events,
             r_on.dead_expert_events, r_off.dead_expert_events),
            ("Final Neff", o_on.final_neff, o_off.final_neff,
             r_on.final_neff, r_off.final_neff),
            ("Mean Neff", o_on.mean_neff, o_off.mean_neff,
             r_on.mean_neff, r_off.mean_neff),
            ("Nonzero Press %", o_on.nonzero_pressure_ratio*100, 0,
             r_on.nonzero_pressure_ratio*100, 0),
        ]

        for name, o_on_v, o_off_v, r_on_v, r_off_v in rows:
            print(f"{name:<20} {o_on_v:<12.2f} {o_off_v:<12.2f} {r_on_v:<12.2f} {r_off_v:<12.2f}")

        # Deltas
        print("\nController benefit (ON - OFF):")
        o_delta = o_on.final_neff - o_off.final_neff
        r_delta = r_on.final_neff - r_off.final_neff
        print(f"  Original: Final Neff +{o_delta:.2f}")
        print(f"  Replication: Final Neff +{r_delta:.2f}")

        if r_delta > 0:
            print("\n✓ REPLICATION CONFIRMS: Controller helps under different stress configuration")
        else:
            print("\n✗ REPLICATION FAILED: Controller did not help under this stress")


def main():
    output_dir = Path("out-replication")

    print("="*70)
    print("REPLICATION: Capacity Stress Variant")
    print("="*70)
    print("\nOriginal: top_k=1, train_capacity=1.25")
    print("This run: top_k=1, train_capacity=1.0 (tighter)")
    print()

    replication = run_replication(
        output_dir=output_dir,
        pretrain_iters=500,
        finetune_iters=300,
        finetune_top_k=1,
        finetune_capacity=1.0,
    )

    # Load original results if available
    original_path = Path("out-domain-shift-stressed/results.json")
    original = {}
    if original_path.exists():
        with open(original_path) as f:
            orig_data = json.load(f)
            for k, v in orig_data.items():
                original[k] = ReplicationMetrics(**v)

    if replication:
        if original:
            print_comparison(original, replication)
        else:
            print("\nOriginal results not found - showing replication only")
            if "ON" in replication and "OFF" in replication:
                r_on, r_off = replication["ON"], replication["OFF"]
                delta = r_on.final_neff - r_off.final_neff
                print(f"Final Neff: ON={r_on.final_neff:.2f}, OFF={r_off.final_neff:.2f}, Delta=+{delta:.2f}")
                print(f"Nonzero pressure: {r_on.nonzero_pressure_ratio:.1%}")

        # Save results
        with open(output_dir / "results.json", 'w') as f:
            json.dump({k: asdict(v) for k, v in replication.items()}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
