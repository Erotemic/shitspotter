# v6 evaluation — TO BE FILLED IN AFTER RUNNING

Fill in immediately after `run.sh` completes. Commit `EVAL.md` in the
same commit that updates `recipe.yaml` if any tuning was needed.

## Headline numbers

| Metric                        | v4 baseline | v6 measured | Δ vs v4 | Verdict      |
|-------------------------------|-------------|-------------|---------|--------------|
| AP @ IoU=0.5 (simplified test)| 0.406       | TBD         | TBD     | TBD          |
| Desktop CPU latency mean (ms) | 17.6        | TBD         | TBD     | TBD          |
| Desktop CPU latency p99 (ms)  | 27.2        | TBD         | TBD     | TBD          |
| Eligibility class             | HOST_PROMISING | TBD      | —       | —            |

## Pivot verdict

- [ ] AP within ±0.01 of v4 → **kit pivot validated**, proceed to v7.
- [ ] AP outside ±0.01 → **investigate** (do not proceed to v7 yet).

## Run identity

- Recipe file SHA: `git rev-parse HEAD -- experiments/mobile_app_training_v6/recipe.yaml`
- Image tag: `<docker image:tag>`
- Kit commit: `<kwcoco_detector_kit HEAD>`
- DEIMv2 commit: `<tpl/DEIMv2 HEAD>`
- Sweep timestamp dir: `/data/joncrall/kcd/v6/sweeps/<TS>`

## Notes
