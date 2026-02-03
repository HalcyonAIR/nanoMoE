#!/usr/bin/env python3
"""
Clock 3 Persistence Test: Does "slow remembers" work?

This is the minimal test for path-dependent decision making:
1. Run stressed fine-tune (Run A) - controller learns what works
2. Persist control state (harm_backoff, mode_scores per layer)
3. Run same stressed fine-tune again (Run B) with persisted state
4. Compare: does Run B need less intervention to achieve same topology?

If Run B has lower mean_scale but similar/better Neff, the controller
"remembered" what it learned and needed less active steering.

This is Clock 3 in embryonic form: prior state constraining future action.
"""

import os
import sys
import json
import pickle
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PersistenceMetrics:
    """Metrics for one persistence test run."""
    run_id: str  # "A" (fresh) or "B" (with prior)

    # Topology outcomes
    final_neff: float
    mean_neff: float
    dead_events: int

    # Intervention effort
    mean_scale: float
    total_scale: float  # Sum of all scales (total "work" done)
    mean_pressure: float
    nonzero_pressure_ratio: float

    # State evolution
    final_backoff_mean: float  # Mean harm_backoff across layers at end
    abstain_ratio: float


@dataclass
class PersistedControlState:
    """Minimal state to persist between runs - the "slow memory"."""
    layer_states: Dict[int, Dict]  # layer_id -> {harm_backoff, mode_scores, pressure}

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self.layer_states, f)

    @classmethod
    def load(cls, path: Path) -> 'PersistedControlState':
        with open(path, 'rb') as f:
            layer_states = pickle.load(f)
        return cls(layer_states=layer_states)


def extract_final_control_state(telemetry_dir: Path) -> PersistedControlState:
    """Extract final control state from telemetry for persistence."""
    decisions_file = telemetry_dir / 'control_decisions.jsonl'

    # Read all decisions and keep last per layer
    final_states = {}

    with open(decisions_file) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            layer_id = d['layer_id']
            final_states[layer_id] = {
                'harm_backoff': d['computed'].get('harm_backoff', 1.0),
                'mode_scores': d['computed'].get('mode_scores', None),
                'pressure': d['computed'].get('pressure', 0.0),
                'active_mode': d['computed'].get('active_mode', 'anti_dominance'),
                # Critical: persist prev_top2 so harm detection works correctly
                'prev_top2': d['observed'].get('top2_share', 0.5),
                'prev_scale': d['actuator'].get('lens_scale', 0.0),
            }

    return PersistedControlState(layer_states=final_states)


def parse_telemetry(telemetry_dir: Path) -> Dict:
    """Parse telemetry for metrics."""
    results = {
        'neffs': [],
        'scales': [],
        'pressures': [],
        'backoffs': [],
        'dead_events': 0,
        'abstain_count': 0,
        'total_decisions': 0,
    }

    decisions_file = telemetry_dir / 'control_decisions.jsonl'
    if decisions_file.exists():
        with open(decisions_file) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                results['neffs'].append(d['observed']['n_effective'])
                results['scales'].append(d['actuator']['lens_scale'])
                results['pressures'].append(d['computed']['pressure'])
                results['backoffs'].append(d['computed'].get('harm_backoff', 1.0))
                results['total_decisions'] += 1

                if d['observed']['dead_expert_count'] > 0:
                    results['dead_events'] += 1
                if d['computed'].get('abstain', False):
                    results['abstain_count'] += 1

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


def inject_prior_state(prior: PersistedControlState):
    """
    Inject prior state into controller initialization.

    This modifies the chronomoe controller to accept initial state.
    For this test, we'll write a state file that train.py can load.
    """
    # Write prior state to a known location
    prior_path = Path("out-clock3/prior_state.pkl")
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior.save(prior_path)
    return prior_path


def run_persistence_test(
    output_dir: Path,
    pretrain_iters: int = 500,
    finetune_iters: int = 300,
) -> Dict[str, PersistenceMetrics]:
    """
    Run the persistence test.

    1. Pre-train on Shakespeare (shared)
    2. Run A: Fresh controller, stressed fine-tune
    3. Extract and persist control state
    4. Run B: Same fine-tune with persisted state as prior
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # Configs
    pretrain_config = '''# Pre-training Config
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

    finetune_base = '''# Fine-tuning Config (STRESSED)
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

    # =========================================================================
    # Phase 1: Pre-train (shared checkpoint)
    # =========================================================================
    print("\n" + "="*70)
    print("CLOCK 3 TEST: Pre-training (shared)")
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

    # =========================================================================
    # Phase 2: Run A - Fresh controller
    # =========================================================================
    print("\n" + "="*70)
    print("RUN A: Fresh controller (no prior)")
    print("="*70)

    run_a_dir = output_dir / "run_A"
    run_a_config = finetune_base + f'''
out_dir = '{run_a_dir}'
dataset = 'tinystories'
use_chrono_controller = True
max_iters = {finetune_iters}
lr_decay_iters = {finetune_iters}
min_lr = 1e-4
init_from = 'finetune'
ckpt_path = '{pretrain_ckpt}'
'''

    if not run_training(run_a_config, run_a_dir, "Run A (fresh)"):
        return results

    # Extract metrics
    run_a_telemetry = list((run_a_dir / "telemetry").glob("run_*"))[0]
    run_a_data = parse_telemetry(run_a_telemetry)

    neffs = run_a_data['neffs']
    scales = run_a_data['scales']
    pressures = run_a_data['pressures']
    backoffs = run_a_data['backoffs']

    results['A'] = PersistenceMetrics(
        run_id='A',
        final_neff=neffs[-1] if neffs else 0,
        mean_neff=float(np.mean(neffs)) if neffs else 0,
        dead_events=run_a_data['dead_events'],
        mean_scale=float(np.mean(scales)) if scales else 0,
        total_scale=float(sum(scales)) if scales else 0,
        mean_pressure=float(np.mean(pressures)) if pressures else 0,
        nonzero_pressure_ratio=sum(1 for p in pressures if p > 0.01) / max(1, len(pressures)),
        final_backoff_mean=float(np.mean(backoffs[-4:])) if len(backoffs) >= 4 else 1.0,  # Last checkpoint
        abstain_ratio=run_a_data['abstain_count'] / max(1, run_a_data['total_decisions']),
    )

    print(f"  Run A: Final Neff={results['A'].final_neff:.2f}, "
          f"Mean Scale={results['A'].mean_scale:.4f}, "
          f"Total Scale={results['A'].total_scale:.2f}")

    # =========================================================================
    # Phase 3: Extract and persist control state
    # =========================================================================
    print("\n" + "="*70)
    print("PERSISTING CONTROL STATE (Clock 3 memory)")
    print("="*70)

    prior = extract_final_control_state(run_a_telemetry)
    prior_path = output_dir / "prior_state.pkl"
    prior.save(prior_path)

    print(f"  Persisted state for {len(prior.layer_states)} layers:")
    for layer_id, state in prior.layer_states.items():
        print(f"    Layer {layer_id}: backoff={state['harm_backoff']:.3f}, "
              f"mode={state['active_mode']}, pressure={state['pressure']:.3f}")

    # =========================================================================
    # Phase 4: Run B - With persisted prior
    # =========================================================================
    print("\n" + "="*70)
    print("RUN B: Controller with persisted prior")
    print("="*70)

    run_b_dir = output_dir / "run_B"
    run_b_config = finetune_base + f'''
out_dir = '{run_b_dir}'
dataset = 'tinystories'
use_chrono_controller = True
max_iters = {finetune_iters}
lr_decay_iters = {finetune_iters}
min_lr = 1e-4
init_from = 'finetune'
ckpt_path = '{pretrain_ckpt}'
chrono_prior_state_path = '{prior_path}'
'''

    if not run_training(run_b_config, run_b_dir, "Run B (with prior)"):
        # If prior loading isn't implemented yet, run without it as baseline
        print("  Note: Prior loading may not be implemented in train.py yet")
        print("  Running as fresh controller for comparison baseline")

        run_b_config_fallback = finetune_base + f'''
out_dir = '{run_b_dir}'
dataset = 'tinystories'
use_chrono_controller = True
max_iters = {finetune_iters}
lr_decay_iters = {finetune_iters}
min_lr = 1e-4
init_from = 'finetune'
ckpt_path = '{pretrain_ckpt}'
'''
        if not run_training(run_b_config_fallback, run_b_dir, "Run B (fallback)"):
            return results

    # Extract metrics
    run_b_telemetry = list((run_b_dir / "telemetry").glob("run_*"))[0]
    run_b_data = parse_telemetry(run_b_telemetry)

    neffs = run_b_data['neffs']
    scales = run_b_data['scales']
    pressures = run_b_data['pressures']
    backoffs = run_b_data['backoffs']

    results['B'] = PersistenceMetrics(
        run_id='B',
        final_neff=neffs[-1] if neffs else 0,
        mean_neff=float(np.mean(neffs)) if neffs else 0,
        dead_events=run_b_data['dead_events'],
        mean_scale=float(np.mean(scales)) if scales else 0,
        total_scale=float(sum(scales)) if scales else 0,
        mean_pressure=float(np.mean(pressures)) if pressures else 0,
        nonzero_pressure_ratio=sum(1 for p in pressures if p > 0.01) / max(1, len(pressures)),
        final_backoff_mean=float(np.mean(backoffs[-4:])) if len(backoffs) >= 4 else 1.0,
        abstain_ratio=run_b_data['abstain_count'] / max(1, run_b_data['total_decisions']),
    )

    print(f"  Run B: Final Neff={results['B'].final_neff:.2f}, "
          f"Mean Scale={results['B'].mean_scale:.4f}, "
          f"Total Scale={results['B'].total_scale:.2f}")

    return results


def print_summary(results: Dict[str, PersistenceMetrics]):
    """Print comparison summary."""
    if 'A' not in results or 'B' not in results:
        print("\nIncomplete results")
        return

    a, b = results['A'], results['B']

    print("\n" + "="*70)
    print("CLOCK 3 PERSISTENCE TEST RESULTS")
    print("="*70)

    print(f"\n{'Metric':<25} {'Run A (fresh)':<15} {'Run B (prior)':<15} {'Delta':<10}")
    print("-"*65)

    rows = [
        ("Final Neff", a.final_neff, b.final_neff),
        ("Mean Neff", a.mean_neff, b.mean_neff),
        ("Dead Events", a.dead_events, b.dead_events),
        ("Mean Scale", a.mean_scale, b.mean_scale),
        ("Total Scale", a.total_scale, b.total_scale),
        ("Abstain Ratio", a.abstain_ratio, b.abstain_ratio),
    ]

    for name, a_val, b_val in rows:
        delta = b_val - a_val
        print(f"{name:<25} {a_val:<15.3f} {b_val:<15.3f} {delta:+.3f}")

    print("\n" + "="*70)
    print("VERDICT: Does 'slow remembers' work?")
    print("="*70)

    # Success criteria:
    # - Similar or better Neff (topology not worse)
    # - Lower total_scale (less intervention needed)
    # - Or higher abstain_ratio (knew when not to intervene)

    neff_ok = b.final_neff >= a.final_neff - 0.05
    less_work = b.total_scale < a.total_scale * 0.9  # 10% less work
    more_abstain = b.abstain_ratio > a.abstain_ratio + 0.05

    if neff_ok and (less_work or more_abstain):
        print("\n✓ YES: Run B achieved similar topology with less intervention")
        print("  The controller 'remembered' what worked and needed less steering.")
        if less_work:
            reduction = (1 - b.total_scale / a.total_scale) * 100
            print(f"  Total intervention reduced by {reduction:.1f}%")
        if more_abstain:
            print(f"  Abstain ratio increased from {a.abstain_ratio:.1%} to {b.abstain_ratio:.1%}")
    elif neff_ok:
        print("\n~ PARTIAL: Similar topology but no reduction in intervention")
        print("  Prior state loaded but didn't change behavior significantly.")
    else:
        print("\n✗ NO: Prior state didn't help or hurt topology")
        print("  May need different persistence strategy or prior isn't being loaded.")


def main():
    output_dir = Path("out-clock3")

    print("="*70)
    print("CLOCK 3 PERSISTENCE TEST")
    print("="*70)
    print("\nTesting: Can the controller 'remember' across runs?")
    print("If Run B needs less intervention than Run A for same topology,")
    print("we have path-dependent decision making - Clock 3 in embryo.")
    print()

    results = run_persistence_test(
        output_dir=output_dir,
        pretrain_iters=500,
        finetune_iters=300,
    )

    if results:
        print_summary(results)

        # Save results
        with open(output_dir / "results.json", 'w') as f:
            json.dump({k: asdict(v) for k, v in results.items()}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
