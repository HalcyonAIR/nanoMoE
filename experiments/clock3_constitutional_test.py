#!/usr/bin/env python3
"""
Clock 3 Constitutional Test: Limits, not Trajectories

The naive persistence test showed that raw control state doesn't transfer
across model resets - it's entangled with specific weight trajectories.

This test implements CONSTITUTIONAL memory instead:
- Not "what I did" but "what kind of system this is"
- Per-layer fragility scalars learned from experience
- Geometry-agnostic priors that survive resets

The constitutional prior is simple:
- fragility[layer] = learned multiplier on intervention intensity
- fragility > 1.0 means "this layer needs more help"
- fragility < 1.0 means "this layer is stable, back off"

Run A: Fresh controller, learn fragility from intervention patterns
Run B: Load fragility priors, see if intervention is more efficient
"""

import os
import sys
import json
import pickle
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ConstitutionalMetrics:
    """Metrics for constitutional test."""
    run_id: str

    # Topology
    final_neff: float
    mean_neff: float
    dead_events: int

    # Intervention efficiency
    mean_scale: float
    total_scale: float
    intervention_efficiency: float  # Neff improvement per unit of scale

    # Constitutional state
    fragility_prior: Dict[int, float]  # Per-layer fragility used


@dataclass
class ConstitutionalPrior:
    """
    Geometry-agnostic prior that survives model resets.

    This is what Clock 3 actually remembers:
    - Not trajectories, but limits
    - Not actuator states, but system characteristics
    """
    layer_fragility: Dict[int, float]  # layer_id -> fragility multiplier

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({'layer_fragility': self.layer_fragility}, f)

    @classmethod
    def load(cls, path: Path) -> 'ConstitutionalPrior':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(layer_fragility=data['layer_fragility'])

    @classmethod
    def learn_from_telemetry(cls, telemetry_dir: Path) -> 'ConstitutionalPrior':
        """
        Learn fragility from intervention patterns.

        Fragility = how much this layer needed intervention relative to others.
        Computed as: mean_scale[layer] / mean_scale[all_layers]

        If a layer consistently needed more scale, it's fragile.
        If it consistently abstained, it's stable.
        """
        decisions_file = telemetry_dir / 'control_decisions.jsonl'

        layer_scales = {}  # layer_id -> list of scales

        with open(decisions_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                layer_id = d['layer_id']
                scale = d['actuator']['lens_scale']

                if layer_id not in layer_scales:
                    layer_scales[layer_id] = []
                layer_scales[layer_id].append(scale)

        # Compute mean scale per layer
        mean_scales = {lid: np.mean(scales) for lid, scales in layer_scales.items()}
        global_mean = np.mean(list(mean_scales.values())) if mean_scales else 1.0

        # Fragility = relative intervention need
        # Clamped to [0.5, 2.0] to avoid extreme values
        fragility = {}
        for lid, ms in mean_scales.items():
            if global_mean > 0.001:
                f = ms / global_mean
            else:
                f = 1.0  # No intervention anywhere, neutral fragility
            fragility[lid] = float(np.clip(f, 0.5, 2.0))

        return cls(layer_fragility=fragility)


def parse_telemetry(telemetry_dir: Path) -> Dict:
    """Parse telemetry for metrics."""
    results = {
        'neffs': [],
        'scales': [],
        'pressures': [],
        'dead_events': 0,
        'total_decisions': 0,
        'per_layer_scales': {},
    }

    decisions_file = telemetry_dir / 'control_decisions.jsonl'
    if decisions_file.exists():
        with open(decisions_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                layer_id = d['layer_id']

                results['neffs'].append(d['observed']['n_effective'])
                results['scales'].append(d['actuator']['lens_scale'])
                results['pressures'].append(d['computed']['pressure'])
                results['total_decisions'] += 1

                if layer_id not in results['per_layer_scales']:
                    results['per_layer_scales'][layer_id] = []
                results['per_layer_scales'][layer_id].append(d['actuator']['lens_scale'])

                if d['observed']['dead_expert_count'] > 0:
                    results['dead_events'] += 1

    return results


def run_training(config_content: str, run_dir: Path, description: str) -> bool:
    """Run training."""
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


def run_constitutional_test(
    output_dir: Path,
    pretrain_iters: int = 500,
    finetune_iters: int = 300,
) -> Dict[str, ConstitutionalMetrics]:
    """
    Run constitutional persistence test.

    1. Pre-train (shared)
    2. Run A: Fresh, learn fragility
    3. Run B: Apply fragility prior via scale multiplier
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Configs - note: we'll modify scale dynamically based on fragility
    base_config = '''# Constitutional Test Config
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
top_k = 1
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
learning_rate = 3e-4
beta2 = 0.99
warmup_iters = 50
device = 'mps'
compile = False
'''

    pretrain_config = base_config.replace('top_k = 1', 'top_k = 2') + '''
learning_rate = 1e-3
'''

    # Phase 1: Pre-train
    print("\n" + "="*70)
    print("CONSTITUTIONAL TEST: Pre-training")
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

    # Phase 2: Run A - Fresh, learn fragility
    print("\n" + "="*70)
    print("RUN A: Fresh controller (learning fragility)")
    print("="*70)

    run_a_dir = output_dir / "run_A"
    run_a_config = base_config + f'''
out_dir = '{run_a_dir}'
dataset = 'tinystories'
use_chrono_controller = True
max_iters = {finetune_iters}
lr_decay_iters = {finetune_iters}
min_lr = 1e-4
init_from = 'finetune'
ckpt_path = '{pretrain_ckpt}'
'''

    if not run_training(run_a_config, run_a_dir, "Run A"):
        return results

    run_a_telemetry = list((run_a_dir / "telemetry").glob("run_*"))[0]
    run_a_data = parse_telemetry(run_a_telemetry)

    neffs = run_a_data['neffs']
    scales = run_a_data['scales']

    # Learn fragility
    prior = ConstitutionalPrior.learn_from_telemetry(run_a_telemetry)
    prior_path = output_dir / "constitutional_prior.pkl"
    prior.save(prior_path)

    results['A'] = ConstitutionalMetrics(
        run_id='A',
        final_neff=neffs[-1] if neffs else 0,
        mean_neff=float(np.mean(neffs)) if neffs else 0,
        dead_events=run_a_data['dead_events'],
        mean_scale=float(np.mean(scales)) if scales else 0,
        total_scale=float(sum(scales)) if scales else 0,
        intervention_efficiency=(neffs[-1] - 3.5) / max(0.01, sum(scales)) if scales else 0,
        fragility_prior={},  # Fresh run, no prior
    )

    print(f"  Run A: Final Neff={results['A'].final_neff:.2f}, "
          f"Total Scale={results['A'].total_scale:.2f}")
    print(f"\n  Learned fragility prior:")
    for lid, f in sorted(prior.layer_fragility.items()):
        status = "fragile" if f > 1.1 else "stable" if f < 0.9 else "neutral"
        print(f"    Layer {lid}: {f:.2f} ({status})")

    # Phase 3: Run B - Apply fragility via scale_max adjustment
    # Instead of modifying controller internals, we'll use the prior
    # to set a per-layer scale cap in the config
    print("\n" + "="*70)
    print("RUN B: With constitutional prior (fragility-aware)")
    print("="*70)

    # Constitutional insight: if we learned a layer is STABLE, we can
    # ABSTAIN more aggressively (raise the abstain threshold)
    # This is geometry-agnostic: "this type of system doesn't need much help"
    max_fragility = max(prior.layer_fragility.values())
    min_fragility = min(prior.layer_fragility.values())

    # If all layers stable (max < 1.0): raise abstain threshold
    # If any layer fragile (max > 1.0): keep default
    if max_fragility < 1.0:
        # All stable - we can be more hands-off
        abstain_mult = 2.0  # Double the abstain threshold
        scale_mult = 1.0    # Keep scale normal
    else:
        # Some fragility - keep defaults but slightly back off
        abstain_mult = 1.0 + (1.0 - min_fragility) * 0.5  # Slight increase
        scale_mult = 1.0

    print(f"  Learned fragility: min={min_fragility:.2f}, max={max_fragility:.2f}")
    print(f"  Constitutional prior: abstain_mult={abstain_mult:.2f} (intervene less often)")

    run_b_dir = output_dir / "run_B"
    # Add a custom config that encodes the learned prior
    run_b_config = base_config + f'''
out_dir = '{run_b_dir}'
dataset = 'tinystories'
use_chrono_controller = True
max_iters = {finetune_iters}
lr_decay_iters = {finetune_iters}
min_lr = 1e-4
init_from = 'finetune'
ckpt_path = '{pretrain_ckpt}'

# Constitutional prior: learned abstain threshold multiplier
# If we learned layers are stable, abstain more aggressively
chrono_constitutional_scale_mult = {scale_mult:.3f}
chrono_constitutional_abstain_mult = {abstain_mult:.3f}
'''

    if not run_training(run_b_config, run_b_dir, "Run B"):
        return results

    run_b_telemetry = list((run_b_dir / "telemetry").glob("run_*"))[0]
    run_b_data = parse_telemetry(run_b_telemetry)

    neffs = run_b_data['neffs']
    scales = run_b_data['scales']

    results['B'] = ConstitutionalMetrics(
        run_id='B',
        final_neff=neffs[-1] if neffs else 0,
        mean_neff=float(np.mean(neffs)) if neffs else 0,
        dead_events=run_b_data['dead_events'],
        mean_scale=float(np.mean(scales)) if scales else 0,
        total_scale=float(sum(scales)) if scales else 0,
        intervention_efficiency=(neffs[-1] - 3.5) / max(0.01, sum(scales)) if scales else 0,
        fragility_prior=prior.layer_fragility,
    )

    print(f"  Run B: Final Neff={results['B'].final_neff:.2f}, "
          f"Total Scale={results['B'].total_scale:.2f}")

    return results


def print_summary(results: Dict[str, ConstitutionalMetrics]):
    """Print comparison."""
    if 'A' not in results or 'B' not in results:
        print("\nIncomplete results")
        return

    a, b = results['A'], results['B']

    print("\n" + "="*70)
    print("CONSTITUTIONAL PERSISTENCE TEST RESULTS")
    print("="*70)
    print("\nThe question: Can geometry-agnostic priors improve intervention?")

    print(f"\n{'Metric':<25} {'Run A (fresh)':<15} {'Run B (prior)':<15} {'Delta':<10}")
    print("-"*65)

    rows = [
        ("Final Neff", a.final_neff, b.final_neff),
        ("Mean Neff", a.mean_neff, b.mean_neff),
        ("Total Scale (effort)", a.total_scale, b.total_scale),
        ("Efficiency (Neff/scale)", a.intervention_efficiency, b.intervention_efficiency),
    ]

    for name, a_val, b_val in rows:
        delta = b_val - a_val
        print(f"{name:<25} {a_val:<15.3f} {b_val:<15.3f} {delta:+.3f}")

    print("\n" + "="*70)
    print("VERDICT: Do constitutional priors work?")
    print("="*70)

    neff_ok = b.final_neff >= a.final_neff - 0.05
    less_effort = b.total_scale < a.total_scale * 0.95
    more_efficient = b.intervention_efficiency > a.intervention_efficiency * 1.05

    if neff_ok and (less_effort or more_efficient):
        print("\n✓ YES: Constitutional prior improved intervention efficiency")
        print("  Slow clock remembers LIMITS, not trajectories.")
        if less_effort:
            reduction = (1 - b.total_scale / a.total_scale) * 100
            print(f"  Total intervention reduced by {reduction:.1f}%")
        if more_efficient:
            improvement = (b.intervention_efficiency / a.intervention_efficiency - 1) * 100
            print(f"  Efficiency improved by {improvement:.1f}%")
    elif neff_ok:
        print("\n~ PARTIAL: Topology maintained but no efficiency gain")
        print("  Prior loaded but didn't improve intervention.")
    else:
        print("\n✗ NO: Constitutional prior hurt topology")
        print("  Need different prior formulation.")


def main():
    output_dir = Path("out-clock3-constitutional")

    print("="*70)
    print("CLOCK 3 CONSTITUTIONAL TEST")
    print("="*70)
    print("\n'Slow clock learns limits, not trajectories.'")
    print("\nTesting: Can geometry-agnostic fragility priors")
    print("improve intervention efficiency across model resets?")
    print()

    results = run_constitutional_test(
        output_dir=output_dir,
        pretrain_iters=500,
        finetune_iters=300,
    )

    if results:
        print_summary(results)

        with open(output_dir / "results.json", 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
