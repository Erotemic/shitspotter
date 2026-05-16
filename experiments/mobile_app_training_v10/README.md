# mobile_app_training_v10 — best-of-everything ship candidates

v10 combines whatever worked in v7-v9 into a single recipe per cell,
trained for longer with EMA, and freezes the ship candidates.

| Cell        | v4 baseline | v10 final | Δ      | Status   |
|-------------|-------------|-----------|--------|----------|
| pico@416    | 0.406       | TBD       | TBD    | TBD      |
| n@640       | 0.520       | TBD       | TBD    | TBD      |

## What goes into the v10 recipe

Fill in `recipe.yaml` **after** v9 is evaluated, copying:

- The winning `train_policy` (multiscale vs fixed) from v7's EVAL.md
- The winning distillation training data path from v9's EVAL.md
- The winning round-count from v8's EVAL.md (if mining helped, run
  v10 as one more round on top of the v9-distilled checkpoint)
- 1.5–2× the epoch count of the best round
- EMA toggled on (the kit's deimv2 trainer exposes this in
  `_train_deimv2_variant.sh`; mirror via a recipe knob if added)

There is no v10 recipe.yaml committed yet — it's intentionally written
after the upstream cells' EVAL.md files are filled in. That keeps v10
honest: it's a synthesis, not a prediction.

## Quick start (inside the docker image)

Same shape as v6/v7/v9:

```bash
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v10/run.sh
```

## Success criterion

Either cell beating its v4 number by **at least +5 AP** is the ship
gate. If only one cell hits +5, ship that one and downgrade the other
to "future work."

## Deliverables for the ship cut

After the v10 numbers land:

1. Copy the winning ONNX into `tpl/shitspotter-phone-app/`'s
   model assets dir per the phone app's `006_adding_a_new_model.md`.
2. Update the phone app's `ModelRegistry` (`DEIMV2_PICO_416` /
   `DEIMV2_N_640`) to reference the new ONNX. The
   modelspec sidecar from the kit's export already has the right
   shape and postprocess params.
3. Commit the EVAL.md, the modelspec, and the recipe in one commit.
4. Tag the shitspotter repo with `mobile-v10-{cell}-{date}`.
