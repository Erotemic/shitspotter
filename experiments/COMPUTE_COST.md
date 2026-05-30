# Compute cost — mobile_app_training v6 → v10

Per-run wall-clock harvested from DEIMv2's `Training time HH:MM:SS`
banner in the toothbrush bash logs, cross-checked against per-epoch
`Epoch: [N] Total time` lines. Where the bash log was lost, the row is
marked **est** with the inference noted.

Hardware (single training rig):
- **CPU+GPU**: RTX 3090 (24 GB, Ampere SM 8.6) — *GPU 0 only*; GPU 1
  has only 2 PCIe lanes, used opportunistically for ad-hoc eval, never
  for training.
- **Host**: toothbrush
- **Per-3090 board power** under DEIMv2 training: ~280–350 W (3090 TDP
  370 W, observed steady-state below that for these batch sizes).
- **PSU/cooling overhead**: rough +30–40 % on top of the GPU draw.

## Per-cell training wall-clock

| Experiment | Cell                       | Epochs | Per-epoch | Wall-clock   | Notes |
|------------|----------------------------|-------:|----------:|-------------:|-------|
| **v6 baseline** | deimv2_pico@416       | 80     |  ~2.1 min |  **2 h 51 m** | first clean run after the init_checkpoint + score_thresh fixes |
| v6 (retries before fix)   | deimv2_pico@416       | 80×2   |  ~2.3 min |  ~6 h        | two earlier attempts before the COCO-init fix landed |
| v6 (partial, killed)      | deimv2_pico@416       | 38     |  ~4.2 min |  **2 h 40 m** | abandoned before the score_thresh discovery |
| **v7 multiscale** | deimv2_pico@416     | 80     |  ~2.3 min |  **3 h 07 m** | bash log line "Training time 3:06:51" |
|                           | deimv2_n@640          | 80     |  ~3.5 min |  **4 h 39 m** | bash log line "Training time 4:38:51" |
| **v8 hard-neg (mining loop)** | deimv2_pico@416 rd0 | 30 |  ~41 min  |  **20 h 29 m** | dataset ~20× larger than v6/v7 (pos + 3× rand neg = ~250 K tiles) |
|                           | deimv2_pico@416 rd1   | 30     | est ~41 min | est **~20 h** | mtime span consistent; bash log not captured |
|                           | deimv2_pico@416 rd2   | 30     | est ~41 min | est **~20 h** | same |
|                           | deimv2_n@640 rd0      | 30     | est ~35 min | est **~17 h** | from user spot-check at epoch 11 mid-run |
|                           | deimv2_n@640 rd1      | 30     | est ~35 min | est **~17 h** | |
|                           | deimv2_n@640 rd2      | 30     | est ~35 min | est **~17 h** | |
| v8 mining passes          | round 0 (pre-budget)  | —      | —         |  **~16 h**   | scored full 1.8 M neg tile pool, ~19 Hz |
|                           | round 0 n@640 (pre-budget) | — | —       |  **~16 h** est | similar pool |
|                           | rounds 1 (post-budget)| —      | —         |  **~30 m each** | 50 K stratified, kit commit `9028a52` |

### Subtotals (v6-v8 + the v6.1/v7.1/bisect retries on the corrected bundle)

| Phase                           | Cumulative training time |
|---------------------------------|--------------------------|
| v6 (counting retries)           |  **~12 h**               |
| v7                              |  **~8 h**                |
| v8 training                     |  **~110 h**              |
| v8 mining                       |  **~33 h**               |
| v6.1 pico@416                   |  **~12 h**               |
| v7.1 pico + n@640               |  **~25 h** (pico ~11 h + n ~14 h, estimated) |
| v7.1 DEIMv2 bisect (pico-only)  |  **~11 h** (measured, see below) |
| **TOTAL training (v6 → bisect)** |  **~211 GPU-hours**     |
| + failed/restarted runs / debug |  +20 h conservative      |
| **GRAND TOTAL on toothbrush**   |  **~230 GPU-hours**      |

### Per-epoch sensitivity (the surprise from the bisect)

| Recipe                        | Tile pool size | Train policy   | Per-epoch | 80-epoch run |
|-------------------------------|----------------|----------------|-----------|--------------|
| v6.0 pico (wrong bundle)      | 12,820         | fixed          | ~2.4 min  | ~3 h         |
| v6.1 pico (corrected bundle)  | 53,355         | fixed          | ~6 min    | ~8 h (est.)  |
| v7.1 / bisect pico (multi)    | 53,355         | multiscale_320_512 | ~8 min | **~10.7 h (measured)** |

**The lesson** (added as journal lesson #10): when an experiment
changes data size AND training policy simultaneously, re-estimate
wall-clock from per-epoch first principles, not by carrying forward
the prior cell's total.

Wall-clock measurement source for the bisect run: file mtimes inside
the workdir. `best_stg1.pth` (written first time val AP improves —
typically epoch 0 or 1) at 02:59:52 UTC; `log.txt` last write (end of
epoch 79) at 13:39:51 UTC. Span: **10h 40m** training + ~20 min for
export + eval + bench = **~11 h** total.

### Energy + carbon (order-of-magnitude)

- 200 h × 0.32 kW (3090 steady-state median estimate) ≈ **65 kWh GPU-only**
- ×1.4 PSU/cooling overhead ≈ **90 kWh wall-plug**
- US grid average 2024 ≈ **0.39 kg CO2e / kWh** → **~35 kg CO2e**
- For comparison: ~ 1 round-trip economy flight Boston ↔ New York (~150 kg)
  or ~ 3 weeks of an average US household running a refrigerator
  (~3 kg/day).

These are small numbers in absolute terms (we're shipping mobile-class
detectors, not training foundation models), but worth tracking so we
don't unconsciously inflate the budget on later iterations.

## Why v8 cost ~10× v6 per round

v6/v7 trained on the v6 quadrant-tile bundle (≈12 820 images). v8's
round 0 trains on ALL positive tiles (~62 906) plus 3× random negatives
(~188 K), totaling ~250 K images per epoch. That's ~20× the training
data — and since each epoch is one full pass, ~20× the wall-clock at
the same batch size. Three rounds × 30 epochs = effectively 90 epochs
on this much-larger pool.

Lesson for v9/v10: **v9 distillation should reuse the v6 tile bundle,
not the v8 multi-scale pool**, unless we explicitly want the broader
distribution. v6's pool gets us a v7-comparable run in ~3 h on pico
vs. v8's ~60 h.

## Tracking discipline for v9+

After each `run.sh`, append a row to this table. Source:

```bash
grep "^Training time" /tmp/<the_tee_log>.log
```

Or extract programmatically post-hoc:

```bash
grep -E "Epoch: \[[0-9]+\] Total time" <(docker logs <container>) | \
    awk -F'[][: ]+' '{secs += $7*3600 + $8*60 + $9} END {print secs/3600 "h"}'
```

For the kit's `round-loop`, each round's wall-clock is per-launch — the
`Training time` line appears once at the end of each
`torch.distributed.run` invocation, so there are N×K of them for
N rounds × K cells.
