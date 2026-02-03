#!/usr/bin/env python3
"""
Domain-Shift Fine-Tuning Test: Does the controller help during domain adaptation?

This is the critical validation experiment for Phase 2.5. Unlike distillation
which tests a specific training pressure, domain shift tests whether the
controller generalizes as a protective envelope under realistic conditions.

Experiment:
1. Train base model on Shakespeare (standard pre-training)
2. Fine-tune on TinyStories (clearly different domain)
3. Compare controller ON vs OFF during fine-tuning
4. Measure topology health throughout adaptation

Key metrics (in order of importance):
- Dead expert events: absolute failures
- Recovery time: steps to regain healthy Neff after domain shock
- Final Neff: equilibrium topology health
- Min Neff: secondary diagnostic only (momentary dips expected)

Possible outcomes:
- Helps → architecture validated beyond distillation
- Fails like bias 10-15 → Phase 3 (new bases) justified
- Abstain dominates → strengthen "do nothing is a policy" story
- No effect → scope narrows
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DomainShiftMetrics:
    """Metrics for one domain-shift run."""
    condition: str  # "ON" or "OFF"

    # Task performance
    final_train_loss: float
    final_val_loss: float

    # Primary metrics (in priority order)
    dead_expert_events: int        # Absolute topology failures
    recovery_steps: int            # Steps to regain Neff > threshold after shock
    final_neff: float              # Equilibrium health

    # Secondary diagnostics
    min_neff: float                # Momentary dips (expected during shift)
    mean_neff: float               # Average health during fine-tuning
    neff_trajectory: List[float]   # For plotting

    # Controller behavior (ON only)
    mean_pressure: float
    max_pressure: float
    mean_scale: float
    abstain_count: int
    abstain_ratio: float
    mode_switches: int             # How often harm guard switched modes


def parse_telemetry(telemetry_dir: Path, is_controller_on: bool, recovery_threshold: float = 2.4) -> Dict:
    """
    Parse telemetry files for metrics.

    Args:
        telemetry_dir: Path to telemetry directory
        is_controller_on: Whether controller was enabled
        recovery_threshold: Neff value considered "recovered" (60% of 4 experts)
    """
    results = {
        'neffs': [],
        'pressures': [],
        'scales': [],
        'dead_events': 0,
        'abstain_count': 0,
        'mode_switches': 0,
        'total_decisions': 0,
        'recovery_step': -1,
        'losses_train': [],
        'losses_val': [],
        'prev_mode': None,
    }

    # Track when we first saw collapse and when we recovered
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
                    results['neffs'].append(neff)
                    results['pressures'].append(data['computed']['pressure'])
                    results['scales'].append(data['actuator']['lens_scale'])
                    results['total_decisions'] += 1

                    if data['observed']['dead_expert_count'] > 0:
                        results['dead_events'] += 1

                    # Count explicit abstentions
                    if data['computed'].get('abstain', False):
                        results['abstain_count'] += 1

                    # Track mode switches
                    current_mode = data['computed'].get('active_mode', 'anti_dominance')
                    if results['prev_mode'] is not None and current_mode != results['prev_mode']:
                        results['mode_switches'] += 1
                    results['prev_mode'] = current_mode

                    # Track recovery
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

                        # Track recovery (use layer 0 as representative)
                        if layer['layer_id'] == 0:
                            if neff < recovery_threshold:
                                saw_collapse = True
                            elif saw_collapse and neff >= recovery_threshold and results['recovery_step'] < 0:
                                results['recovery_step'] = step

    return results


def parse_training_log(run_dir: Path) -> Tuple[float, float]:
    """Parse final losses from training output."""
    train_loss = 0.0
    val_loss = 0.0

    # Look for loss values in telemetry
    telemetry_dirs = list((run_dir / "telemetry").glob("run_*"))
    if telemetry_dirs:
        snapshots_file = telemetry_dirs[0] / 'snapshots.jsonl'
        if snapshots_file.exists():
            with open(snapshots_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Take last seen values
                        if 'train_loss' in data:
                            train_loss = data['train_loss']
                        if 'val_loss' in data:
                            val_loss = data['val_loss']

    return train_loss, val_loss


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
        print(f"    FAILED: {result.stderr[:500]}")
        return False

    return True


def run_domain_shift_experiment(
    output_dir: Path,
    pretrain_iters: int = 500,
    finetune_iters: int = 300,
) -> Dict[str, DomainShiftMetrics]:
    """
    Run full domain-shift experiment.

    Phase 1: Pre-train on Shakespeare (no controller, establish baseline)
    Phase 2: Fine-tune on TinyStories with controller ON vs OFF
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Common config base (small model for quick iteration)
    base_config = f'''# Domain Shift Test Config
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

    # =========================================================================
    # Phase 1: Pre-train on Shakespeare
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Pre-training on Shakespeare")
    print("="*70)

    pretrain_dir = output_dir / "pretrain_shakespeare"
    pretrain_config = base_config + f'''
out_dir = '{pretrain_dir}'
dataset = 'shakespeare_char'
use_chrono_controller = False
max_iters = {pretrain_iters}
lr_decay_iters = {pretrain_iters}
min_lr = 1e-4
'''

    if not run_training(pretrain_config, pretrain_dir, "Pre-training on Shakespeare"):
        print("Pre-training failed!")
        return results

    # Verify checkpoint exists
    pretrain_ckpt = pretrain_dir / "ckpt.pt"
    if not pretrain_ckpt.exists():
        print(f"Pre-train checkpoint not found at {pretrain_ckpt}")
        return results

    print(f"  Pre-train checkpoint saved: {pretrain_ckpt}")

    # =========================================================================
    # Phase 2: Fine-tune on TinyStories (domain shift)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 2: Fine-tuning on TinyStories (Domain Shift)")
    print("="*70)
    print(f"  Source domain: Shakespeare (character-level)")
    print(f"  Target domain: TinyStories (children's stories)")
    print(f"  Pre-trained model: {pretrain_ckpt}")

    for condition in ["OFF", "ON"]:
        print(f"\n--- Fine-tuning with Controller {condition} ---")

        finetune_dir = output_dir / f"finetune_{condition}"
        finetune_config = base_config + f'''
out_dir = '{finetune_dir}'
dataset = 'tinystories'
use_chrono_controller = {condition == "ON"}
max_iters = {finetune_iters}
lr_decay_iters = {finetune_iters}
min_lr = 1e-4

# Fine-tuning: load weights but reset training state
init_from = 'finetune'
ckpt_path = '{pretrain_ckpt}'

# Lower learning rate for fine-tuning
learning_rate = 3e-4
'''

        if not run_training(finetune_config, finetune_dir, f"Fine-tuning (controller {condition})"):
            print(f"  Fine-tuning {condition} failed!")
            continue

        # Parse results
        telemetry_dirs = list((finetune_dir / "telemetry").glob("run_*"))
        if not telemetry_dirs:
            print(f"  No telemetry found for {condition}")
            continue

        telemetry = parse_telemetry(telemetry_dirs[0], condition == "ON")
        train_loss, val_loss = parse_training_log(finetune_dir)

        # Compute metrics
        neffs = telemetry['neffs'] if telemetry['neffs'] else [0]
        pressures = telemetry['pressures'] if telemetry['pressures'] else [0]
        scales = telemetry['scales'] if telemetry['scales'] else [0]

        # Recovery step: -1 means never recovered, convert to max steps
        recovery_steps = telemetry['recovery_step']
        if recovery_steps < 0:
            recovery_steps = finetune_iters  # Never recovered

        metrics = DomainShiftMetrics(
            condition=condition,
            final_train_loss=train_loss,
            final_val_loss=val_loss,
            dead_expert_events=telemetry['dead_events'],
            recovery_steps=recovery_steps,
            final_neff=neffs[-1] if neffs else 0,
            min_neff=min(neffs) if neffs else 0,
            mean_neff=float(np.mean(neffs)) if neffs else 0,
            neff_trajectory=neffs[:50],  # First 50 for plotting
            mean_pressure=float(np.mean(pressures)) if pressures else 0,
            max_pressure=float(max(pressures)) if pressures else 0,
            mean_scale=float(np.mean(scales)) if scales else 0,
            abstain_count=telemetry['abstain_count'],
            abstain_ratio=telemetry['abstain_count'] / max(1, telemetry['total_decisions']),
            mode_switches=telemetry['mode_switches'],
        )

        results[condition] = metrics

        print(f"  Results for {condition}:")
        print(f"    Dead expert events: {metrics.dead_expert_events}")
        print(f"    Recovery steps: {metrics.recovery_steps}")
        print(f"    Final Neff: {metrics.final_neff:.2f}")
        print(f"    Min Neff: {metrics.min_neff:.2f} (secondary)")
        if condition == "ON":
            print(f"    Mean pressure: {metrics.mean_pressure:.3f}")
            print(f"    Max pressure: {metrics.max_pressure:.3f}")
            print(f"    Mean scale: {metrics.mean_scale:.3f}")
            print(f"    Abstain ratio: {metrics.abstain_ratio:.1%}")
            print(f"    Mode switches: {metrics.mode_switches}")

    return results


def print_summary(results: Dict[str, DomainShiftMetrics]):
    """Print comparison summary with verdict."""
    if "ON" not in results or "OFF" not in results:
        print("\nIncomplete results - cannot compare")
        return

    on = results["ON"]
    off = results["OFF"]

    print("\n" + "="*70)
    print("DOMAIN-SHIFT FINE-TUNING RESULTS")
    print("Shakespeare → TinyStories")
    print("="*70)

    print(f"\n{'Metric':<25} {'Controller ON':<15} {'Controller OFF':<15} {'Delta':<10}")
    print("-"*70)

    # Primary metrics first
    print("PRIMARY METRICS:")
    primary_metrics = [
        ("Dead Expert Events", on.dead_expert_events, off.dead_expert_events, False),
        ("Recovery Steps", on.recovery_steps, off.recovery_steps, False),
        ("Final Neff", on.final_neff, off.final_neff, True),
    ]

    for name, on_val, off_val, higher_better in primary_metrics:
        delta = on_val - off_val
        indicator = ""
        if higher_better:
            indicator = " ✓" if delta > 0 else " ✗" if delta < -0.1 else ""
        else:
            indicator = " ✓" if delta < 0 else " ✗" if delta > 0 else ""
        print(f"  {name:<23} {on_val:<15.2f} {off_val:<15.2f} {delta:+.2f}{indicator}")

    print("\nSECONDARY METRICS:")
    secondary_metrics = [
        ("Min Neff", on.min_neff, off.min_neff),
        ("Mean Neff", on.mean_neff, off.mean_neff),
    ]

    for name, on_val, off_val in secondary_metrics:
        delta = on_val - off_val
        print(f"  {name:<23} {on_val:<15.2f} {off_val:<15.2f} {delta:+.2f}")

    print(f"\nCONTROLLER BEHAVIOR:")
    print(f"  Mean pressure: {on.mean_pressure:.3f}")
    print(f"  Max pressure:  {on.max_pressure:.3f}")
    print(f"  Mean scale:    {on.mean_scale:.3f}")
    print(f"  Abstain ratio: {on.abstain_ratio:.1%} ({on.abstain_count} events)")
    print(f"  Mode switches: {on.mode_switches}")

    # =========================================================================
    # VERDICT
    # =========================================================================
    print("\n" + "="*70)
    print("VERDICT: What does domain shift tell us?")
    print("="*70)

    dead_benefit = off.dead_expert_events - on.dead_expert_events
    recovery_benefit = off.recovery_steps - on.recovery_steps
    neff_benefit = on.final_neff - off.final_neff

    # Determine outcome
    if dead_benefit > 0:
        outcome = "HELPS"
        reason = f"Prevented {dead_benefit} dead expert event(s)"
    elif dead_benefit == 0 and recovery_benefit > 10:
        outcome = "HELPS"
        reason = f"Faster recovery by {recovery_benefit} steps"
    elif dead_benefit == 0 and neff_benefit > 0.1:
        outcome = "HELPS"
        reason = f"Better final topology (Neff +{neff_benefit:.2f})"
    elif on.abstain_ratio > 0.3:
        outcome = "ABSTAINS"
        reason = f"Correctly abstained {on.abstain_ratio:.0%} of time"
    elif dead_benefit < 0:
        outcome = "HURTS"
        reason = f"Caused {-dead_benefit} additional dead expert event(s)"
    elif neff_benefit < -0.2:
        outcome = "HURTS"
        reason = f"Worse final topology (Neff {neff_benefit:.2f})"
    else:
        outcome = "NEUTRAL"
        reason = "No significant effect detected"

    print(f"\nOutcome: {outcome}")
    print(f"Reason: {reason}")

    # Implications for Phase 3
    print("\n" + "-"*70)
    print("IMPLICATIONS:")

    if outcome == "HELPS":
        print("  → Architecture validated beyond distillation")
        print("  → Phase 3 (new bases) is OPTIONAL enhancement")
        print("  → Controller generalizes as protective envelope")
    elif outcome == "HURTS":
        print("  → Failure pattern matches bias=10-15 intermediate regime")
        print("  → Phase 3 (new bases) is JUSTIFIED")
        print("  → Need orthogonal steering directions for domain shift")
    elif outcome == "ABSTAINS":
        print("  → 'Do nothing is a policy' validated under real stress")
        print("  → Harm guard correctly detects domain shift as dangerous")
        print("  → Phase 3 would let controller help instead of abstain")
    else:
        print("  → Scope narrows: controller may only help specific scenarios")
        print("  → Domain shift may not need MoE protection")
        print("  → Consider whether this is a feature or limitation")


def save_results(results: Dict[str, DomainShiftMetrics], output_dir: Path):
    """Save results to JSON."""
    results_dict = {}
    for k, v in results.items():
        d = asdict(v)
        # neff_trajectory can be large, truncate for JSON
        d['neff_trajectory'] = d['neff_trajectory'][:20]
        results_dict[k] = d

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_dict, f, indent=2)


def main():
    """Run domain-shift experiment."""
    output_dir = Path("out-domain-shift")

    print("="*70)
    print("DOMAIN-SHIFT TEST: Controller Under Fine-Tuning Pressure")
    print("="*70)
    print("\nThis experiment tests whether the controller helps during")
    print("domain adaptation from Shakespeare to TinyStories.")
    print()
    print("This is the critical validation for Phase 2.5:")
    print("  - Helps → architecture validated, Phase 3 optional")
    print("  - Hurts → Phase 3 justified (need new bases)")
    print("  - Abstains → harm guard works, Phase 3 would unlock potential")
    print()

    results = run_domain_shift_experiment(
        output_dir=output_dir,
        pretrain_iters=500,
        finetune_iters=300,
    )

    if results:
        print_summary(results)
        save_results(results, output_dir)
        print(f"\nResults saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
