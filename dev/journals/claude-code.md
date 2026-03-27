## 2026-03-27 19:00:00 +0000

Summary of user intent: get up to speed with the ShitSpotter project and its
current research state, then extend the small-data-tuning benchmark to support
OpenGroundingDINO (DINOv2) hyperparameter tuning — specifically batch size and
learning rate sweeps — so we can test whether DINOv2 is as sensitive to these
knobs as DEIMv2 (DINOv3) has been.

### Context and motivation

The small-data benchmark has been the primary comparison harness for DINOv2 vs
DINOv3 detectors. DEIMv2 (DINOv3 backbone) has been extensively tuned through
multiple config-tag sweeps, going from batch-24 down to batch-4, with various
LR scale combinations. The finding is that batch size is a major lever for
DEIMv2, but even at batch-4, DEIMv2 still trails OpenGroundingDINO (DINOv2) by
about 0.10 AP on test.

The critical gap in the experimental design: OpenGroundingDINO has only ever
been run with its default config (batch_size=4, lr=0.0001, lr_backbone=1e-5,
15 epochs). We have no evidence about whether DINOv2 is robust or sensitive to
the same hyperparameter changes. If DINOv2 degrades with larger batches
(like DINOv3 does), that's interesting. If DINOv2 is stable, it strengthens the
conclusion that the DINOv3 gap is architectural/recipe-level rather than purely
a tuning artifact. Either way, it makes the comparison more honest.

### What I did

Extended `run_opengroundingdino_dino_detector_benchmark.sh` to accept three
optional fields in `GDINO_CONFIG_SPECS`: `batch_size`, `lr_scale`, and
`backbone_lr_scale`. The design mirrors the DEIMv2 approach:

- LR scales linearly with batch size from known base values
- Scale factors are additional multipliers on top of that
- Config overrides are appended to the copied Python config template
  (OpenGroundingDINO configs are Python files where later definitions override
  earlier ones)
- The run manifest now records all resolved hyperparameters
- Existing 5-field specs work unchanged (backward compatible)

Also updated the README and created a CHANGELOG.md in the experiment directory
so other agents (including Codex) know what changed and why.

### Design decisions and tradeoffs

1. **Appending overrides to the Python config** rather than templating a new
   config from scratch. This is the simplest approach and matches how the
   existing script already adds `use_coco_eval = False` and `label_list`.
   Risk: if OpenGroundingDINO's config loading changes to not support
   re-definition, this would break. But that's unlikely for a Python config.

2. **Linear LR scaling with batch size by default**, matching the DEIMv2
   convention. The user can set lr_scale=0.5 to keep the absolute LR constant
   when doubling batch size, which is a useful control experiment.

3. **Keeping the default GDINO_CONFIG_SPECS as the single baseline** rather
   than adding new configs to the default. This ensures the existing codex
   agent's workflow is not disrupted. New configs are opt-in via env var.

4. **No analyzer changes needed** — the existing code already reads
   `train_batch_size` from manifests and has fallback parsing of the Python
   config.

### What I verified

- Read the actual deployed OpenGroundingDINO config from a previous baseline
  run on disk at `/data/joncrall/dvc-repos/shitspotter_expt_dvc/...` to
  confirm the base hyperparameter values.
- Confirmed the analyzer handles both old (4-level) and new (5-level) path
  layouts and reads batch size from manifests.
- Verified OpenGroundingDINO has a BERT text encoder (language grounding) while
  DEIMv2 does not — this is a meaningful architectural difference that may
  partly explain the performance gap, especially on small data where language
  priors help.

### What's uncertain

- Whether the OpenGroundingDINO repo is available on the user's training
  machine. The script changes are ready but can't be tested without it.
- Whether OpenGroundingDINO's training loop handles batch_size changes
  gracefully (e.g., does it need gradient accumulation for large batches, or
  will it OOM?).
- The right set of configs to actually run. Suggested starting point:
  baseline (batch 4), batch 8, batch 16, and a batch-8-no-lr-scale control.

### Reusable takeaways

- When one arm of a comparison has been heavily tuned and the other hasn't,
  the gap measurement is confounded. Tuning the baseline is as important as
  tuning the challenger.
- Config-tag systems pay for themselves quickly once you need more than one
  recipe per model family.
- Backward compatibility in config spec formats (optional trailing fields with
  defaults) avoids breaking existing workflows while enabling new experiments.
