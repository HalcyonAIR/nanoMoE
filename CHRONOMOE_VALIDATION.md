# ChronoMoE Phase 2.5 Validation

**Date:** 2026-02-03
**Status:** Protective envelope validated

## Summary

The ChronoMoE controller's "protective envelope" was validated under realistic capacity stress induced by top_k reduction during domain-shift fine-tuning.

## Experiment: Stressed Domain Shift

- **Pre-training:** Shakespeare, top_k=2 (standard routing)
- **Fine-tuning:** TinyStories, top_k=1 (capacity stress)

This creates genuine routing constraint: the model learned to distribute load across 2 experts but must now funnel through 1.

## Results

| Metric | Controller ON | Controller OFF | Delta |
|--------|---------------|----------------|-------|
| Dead Expert Events | 0 | 0 | 0 |
| Final Neff | 3.85 | 3.72 | **+0.13** |
| Mean Neff | 3.82 | 3.72 | +0.10 |

**Controller behavior:**
- Nonzero pressure frequency: 73.1% (stress engaged)
- Mean scale: 0.019 (gentle intervention)
- Abstain ratio: 26.9% (appropriate restraint)

## Verdict

**HELPS** - Controller improves topology health under realistic stress without causing harm.

## Implications

- Protective envelope validated beyond synthetic bias injection
- Phase 3 (orthogonal bases) is optional enhancement, not required
- Controller generalizes to capacity-constrained fine-tuning scenarios

## Files

- `experiments/domain_shift_stressed.py` - Stressed experiment harness
- `experiments/domain_shift_test.py` - Unstressed baseline (validates abstain policy)
- `data/tinystories/prepare.py` - Dataset preparation (Shakespeare vocab aligned)
