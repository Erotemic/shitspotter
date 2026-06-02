# mobile_app_training_v11 — pico @ 640, resolution + distillation (two arms)

v11 chases the **small-poop** failure mode with two isolated levers, run as
separate arms so each is attributable (the v10 lesson: bundling many changes
made nothing attributable, and n@640 silently regressed under the bundle):

| Arm        | Model        | Train data                          | Isolates            |
|------------|--------------|-------------------------------------|---------------------|
| `baseline` | pico @ 640   | v6.1 human GT                       | **resolution** (vs v10 pico@416 = 0.478) |
| `distill`  | pico @ 640   | v6.1 human GT + OGDino teacher pseudo-GT | **distillation** (vs the v11 baseline) |

Everything except the variable under test is held fixed across the arms
(input 640, `multiscale_512_768` policy, COCO-pretrained init, 120 epochs,
same lr). So `baseline − v10pico416` = resolution effect, and
`distill − baseline` = distillation effect.

## Why these levers

- **Resolution.** pico's *native* DEIMv2 config is 640; v10 ran it downscaled
  to 416. "pico" is the architecture (HGNetv2 `Pico` + `LiteEncoder`),
  independent of input size — so 640 is a pure resolution change, giving the
  detector more pixels on small poops. Desktop has the budget (pico@416 was
  14 ms / Pixel 5 7 FPS; ~640 ≈ 34 ms desktop, est. ~3 FPS device — still
  above the 1 FPS floor, to be confirmed by the device bench).
- **Distillation (offline pseudo-GT).** The v9 OpenGroundingDINO bbox teacher
  (AP=0.766 on simplified test GT) labels the same training tiles; its boxes
  are merged into the human GT (tagged `from_teacher=True`, weight 1.0). No
  teacher-forward at train time, no soft labels — value is **recall**: boxes
  the human GT missed, expected to concentrate on small poops. This reuses
  v9's pipeline, which was scaffolded but **never actually run/evaluated**
  (its merged bundle was never produced) — v11 is the first real test.

## Calibrated expectations

- The 0.766 is the *teacher's* capacity (a large DINO transformer).
  Distillation transfers labels, not capacity — pico is tiny. **Do not expect
  pico to approach 0.7.** v9's own success criterion was only +2 AP on pico.
- Resolution is the higher-confidence lever; distillation is the uncertain
  one. A realistic hope: 640 gives a solid bump on small poops, distillation
  adds a few more AP of recall on top. Either could also be a wash — that's
  why they're separate arms.
- **Risk:** the teacher can over-predict on tiles, injecting noisy pseudo-GT.
  Check the merged annotation distribution (count/size/aspect vs human) before
  trusting the distill arm — see EVAL.md.

## Open caveat carried from v10

v10's pico cell was labeled `multiscale_320_512` but the resolved config had
`multiscale_stop_epoch: 1` (multi-scale jitter off after epoch 1 → effectively
fixed 416). v11 uses `multiscale_512_768`; **verify the generated config
actually varies scale** (`runs/<cell>/generated_configs/` → collate
`stop_epoch`) — if it collapses to fixed 640 again, that's fine for a clean
resolution comparison, but note it so "multiscale helped" is not over-claimed.

## Run (inside the shitspotter docker image, GPU host)

```bash
# both arms (baseline first, then teacher pseudo-label + distill):
bash experiments/mobile_app_training_v11/run.sh both

# or one at a time:
bash experiments/mobile_app_training_v11/run.sh baseline
bash experiments/mobile_app_training_v11/run.sh distill     # builds the merged bundle if absent

# dry-run / force-retrain flags pass through to recipe-run:
bash experiments/mobile_app_training_v11/run.sh baseline --dry_run
```

Artifacts land on the SSD: `/media/joncrall/flash1/kcd-ssd/v11/{baseline,distill}/`.

## Prerequisites

- v6.1 tiles staged on the SSD at `/media/joncrall/flash1/kcd-ssd/v6_1/data/`
  (already present — v10 trained from there).
- For the distill arm: the v9 OGDino teacher artifacts — the package template
  `experiments/foundation_detseg_v3/packages/opengroundingdino_sam2_default.yaml`
  and `…/foundation_detseg_v3/v9/selected_detector_checkpoint.yaml` (both
  present), plus the OGDino MSDeformAttention extension built into the image.
- pico COCO init: `/data/joncrall/shitspotter_v4/pretrained/deimv2/deimv2_pico_coco.pth`.

See [EVAL.md](EVAL.md) for the scorecard (incl. size-stratified AP) and
decision criteria.
