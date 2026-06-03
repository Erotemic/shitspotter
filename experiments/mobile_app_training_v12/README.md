# mobile_app_training_v12 — pico @ 768 (resolution push)

The highest-confidence lever in the whole ladder has been **input resolution**:

| Cell                | AP@0.5 | Δ        |
|---------------------|--------|----------|
| v10 pico@416        | 0.478  | —        |
| v11 baseline pico@640 | 0.588 | +0.110  |
| **v12 pico@768**    | TBD    | TBD vs 640 |

v12 takes the same pico architecture one resolution step further (640 → 768) to
test whether more pixels keep paying off on small poops. Everything else is held
identical to the v11 baseline (v6.1 corrected human-GT tiles, COCO-pretrained
init, 120 epochs, same lr), so the AP delta is attributable to resolution.

## Why this and not distillation

The v11 **distill** arm got stuck in a deep OGDino-integration/dep yak-shave
(the kit predictor was written for the wrong groundingdino module layout; see
v11/EVAL.md). v12 deliberately avoids all of that: it is **plain DEIMv2** — no
teacher, no OGDino, no `supervision`/`cv2`, no docker. It trains with exactly
the setup that produced the v11 baseline (0.588) on the host.

## Run (host, uvpy env — no docker needed)

```bash
bash experiments/mobile_app_training_v12/run.sh
# or validate first:
bash experiments/mobile_app_training_v12/run.sh --dry_run
```

Artifacts land at `/media/joncrall/flash1/kcd-ssd/v12/`. The winner manifest is
`/media/joncrall/flash1/kcd-ssd/v12/manifest.json`.

## Latency budget

768 ≈ 1.44× the 640 compute → est. desktop ~47 ms (v11@640 was 32.4 ms), still
under the 80 ms gate. Pixel 5 ≈ ~2 FPS (est.) — above the 1 FPS floor, confirm
with the device bench.

## After it runs

- Fill [EVAL.md](EVAL.md), including **size-stratified AP** via
  `experiments/size_stratified_eval.py` (the whole point is small poops —
  v12 vs v11@640 vs v10@416).
- If 768 beats 640, register it in the app like the v11 640 entry and bench on
  the Pixel 5. If it plateaus/regresses or busts the latency gate, 640 stays
  the ship model and the next lever is **tiled inference** (EVALUATION_ROADMAP #5),
  not more single-pass resolution.
