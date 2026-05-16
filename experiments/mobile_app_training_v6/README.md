# mobile_app_training_v6 — kit-driven baseline reproduction

v6 is the **pivot validation** experiment. It does exactly one thing:
re-train the v4 `deimv2_pico` @ 416×416 fixed-policy cell using
[`kwcoco_detector_kit`](https://github.com/Erotemic/kwcoco-detector-kit)
as the training interface, and confirm the resulting AP matches v4's
**AP=0.406** on the v9 simplified test GT to within ±0.01.

No new technique. No new model. No new data. The only change is
**where the training code lives** (the kit, not bespoke shitspotter
scripts). If v6 doesn't land on the v4 number, we fix the kit before
moving on to v7-v10.

## Layout

| File          | Purpose |
|---------------|---------|
| `recipe.yaml` | Single source of truth for *what* gets trained (matrix, hyperparameters, paths, eligibility). |
| `run.sh`      | Driver: tile the raw splits once, then `kwcoco-detector-kit recipe-run`. |
| `EVAL.md`     | Filled in **after** the run with measured AP, latency, eligibility class, and the AP delta vs v4. |

There is no Python in this directory. All model code lives in the kit.

## Quick start (inside the shitspotter docker image)

```bash
# Build the image (one-time, ~20 min on a 3090):
cd ~/code/shitspotter
python ./dockerfiles/setup_staging.py
export REPO_GIT_HASH=$(git rev-parse --short=12 HEAD)
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t shitspotter:latest \
    --build-arg REPO_GIT_HASH=$REPO_GIT_HASH \
    --build-arg TORCH_CUDA_ARCH_LIST="8.6" \
    -f ./dockerfiles/shitspotter.dockerfile .

# Sanity: GPU + kit health check.
docker run --gpus=all --rm shitspotter:latest \
    kwcoco-detector-kit check-env --runtime --groups core,onnx,deimv2

# Run v6 (single 3090, ~3-4 h end-to-end).
docker run --gpus=all -it --rm \
    -v /data/joncrall/dvc-repos/shitspotter_dvc:/data/joncrall/dvc-repos/shitspotter_dvc:ro \
    -v /data/joncrall/dvc-repos/shitspotter_expt_dvc:/data/joncrall/dvc-repos/shitspotter_expt_dvc:ro \
    -v /data/joncrall/kcd:/data/joncrall/kcd \
    shitspotter:latest \
    bash experiments/mobile_app_training_v6/run.sh
```

## What success looks like

A line at the end of the run that reads (approximately):

```
[recipe] shitspotter_v6_pico416_baseline complete.
[recipe] manifest -> /data/joncrall/kcd/v6/manifest.tsv
```

with the manifest row showing:

| field             | expected         |
|-------------------|------------------|
| candidate_id      | `deimv2_pico_416x416_fixed` |
| test_ap           | 0.396 – 0.416    |
| desktop_latency_ms_mean | ~18 ms (3090 host) |
| eligibility_class | `HOST_PROMISING` |

`test_ap` outside ±0.01 of v4's 0.406 is the trigger to dig in before
running v7. Note the trained kit checkpoint won't be **bit-identical**
to the v4 checkpoint — DETR-style training has nondeterministic
assignment + dataloader ordering. Run-to-run AP variance ~±0.3 even
under the same recipe; ±0.01 of v4 is the gate, not bit equality.

## Where things go

```
/data/joncrall/kcd/v6/
├── data/
│   ├── train_tile_g2.kwcoco.zip       # produced by run.sh's tile step
│   └── vali_tile_g2.kwcoco.zip
├── runs/
│   └── deimv2_pico_416x416_fixed/
│       ├── best_stg2.pth              # trained checkpoint
│       ├── policy.json                # what the kit actually trained
│       ├── generated_configs/train.yml
│       └── export/
│           ├── deimv2_pico_h416_w416.onnx
│           ├── deimv2_pico_h416_w416.modelspec.json
│           └── deimv2_pico_h416_w416.bench.json
├── eval/
│   └── deimv2_pico_416x416_fixed/
│       └── eval/detect_metrics.json
├── sweeps/<timestamp>/index.tsv       # per-cell stage status
├── manifest.tsv                       # eligibility manifest (the row to read)
└── manifest.json                      # richer JSON form
```

## After the run — fill in `EVAL.md`

The kit prints the manifest path and the winner row to stdout. Copy
the AP and latency numbers into `EVAL.md`, note whether the cell
landed in `HOST_PROMISING`, and write the one-sentence verdict on
whether the pivot is validated. Then commit, and proceed to v7.

If AP misses by more than 0.01, **do not** start v7. File the gap
in `EVAL.md`, dig into the kit/v4 divergence (likely culprits:
DEIMv2 SHA drift, tile parameter mismatch, num_top_queries clamp),
and re-run v6.
