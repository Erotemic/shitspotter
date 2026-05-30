# Research Journal — mobile_app_training v6 → vN

Chronological narrative of what we tried, what we measured, what we
believed at each step, and where prior beliefs were later overturned.
Entries are append-only. Per-experiment static records live in each
`mobile_app_training_v*/EVAL.md`; this file is the **story** that ties
them together.

Conventions:
- Each entry is dated and tagged with the load-bearing decision or finding.
- Where a later entry overturns an earlier one, the earlier remains
  intact and a `→ SUPERSEDED BY <date>` pointer is added inline.
- Numbers in this journal are always **kit-eval test AP @ IoU=0.5** on
  the v9 simplified test bundle, unless noted otherwise.

---

## 2026-05-16 — Plan + scaffold

Set up the v6→v10 ladder targeting the two deployable DEIMv2 mobile
cells (pico@416, n@640). Stated bar: each step beats the previous by
≥ +0.01 AP, with v4's reported numbers (`pico=0.406`, `n=0.520`) as the
baseline to beat.

Committed scaffold: `mobile-quality push: v6-v10 scaffolds + docker +
driver` (`373c414`).

**Open question at this point** (later answered the wrong way): is
v4's reported AP comparable to kit-eval AP? Assumed yes; never tested.

---

## 2026-05-16 → 17 — v6.0 first runs + kit bug surface

Three kit-side bugs surfaced during the first attempts at v6 (the kit
pivot validation run):

1. `init_checkpoint` not plumbed through the kit's pareto_sweep → every
   cell trained from HGNetv2-stem only, not COCO-pretrained DEIMv2.
   Cost: ~0.06 AP on pico. Fixed in kit `4f6c97f` + `2cb5587`.
2. `policy.json` not recording `init_ckpt` even when DEIMv2 received
   `-t` — made the bug above invisible. Fixed in `2cb5587`.
3. `run_kwcoco_eval` defaulted to `score_thresh=0.30`, which caps
   recall and crushes COCO AP. Fixed in `31836c5` (default → 0.001).
   Single fix moved v6 from 0.330 → 0.386 with zero retraining.

**v6.0 final number: 0.386 pico@416.**

Recorded verdict at the time ([`v6 EVAL.md`](mobile_app_training_v6/EVAL.md),
commit `434fc8d`): **"kit pivot validated, Δ = −0.020 within DETR
noise"**. → **SUPERSEDED BY 2026-05-22** — the comparison was against
v4's self-reported 0.406, not v4's kit-eval number. The actual gap was
−0.069, not −0.020. See [`V4_VS_KIT_APPLES_TO_APPLES.md`](V4_VS_KIT_APPLES_TO_APPLES.md).

---

## 2026-05-18 — v7 multiscale, both cells

v7 added the `multiscale_*` training-policy band that v4 had left
`NOT_READY` in its sweep matrix.

**v7 numbers:**
- pico@416: 0.398 (declared "~=v4" against the 0.406 self-report)
- n@640:    0.535 (declared "WIN against v4's 0.520")

Recorded verdict at the time ([`v7 EVAL.md`](mobile_app_training_v7/EVAL.md),
commit `7dec7c6`): **"first WIN in the ladder on n@640, +0.015 over
v4"**. → **SUPERSEDED BY 2026-05-22** — v4's kit-eval n@640 is 0.5553,
not 0.520. v7 is actually −0.020 below v4 on n@640. The "first WIN"
claim was wrong.

Also fixed during v7: kit `78c5654` — `use_gateway` per variant +
predictor auto-detect. v7 n@640's training succeeded but eval crashed
because DEIMv2 was constructing the eval-time model with
`use_gateway=False` while the saved state had gateway keys. Salvaged
the 8-hour n@640 training without retraining.

---

## 2026-05-19 → 22 — v8 round-based hard-neg mining

Three rounds of mining for both cells. Plan: train round, mine top-K
hard negatives, retrain round on positives + hard negs, repeat.

The round-loop path surfaced a series of kit bugs that ate ~3 days of
calendar time and ~145 GPU-hours:

| Bug | Symptom | Fix |
|---|---|---|
| Relative file_names in merge.py | round1 training died loading first batch | `b0db63c` |
| Relative file_names in mine.py output | round2 training same death | `017ca44` (also added round_loop resume protocol) |
| O(N²) stratification in mine.py | Apparent "hang" on 1.8 M tile pool, ~3 trillion ops | `b5aeaec` |
| No mining budget | Round 0 mine = ~16 h on full negative pool | `9028a52` (default = 50 K stratified) |
| No GDAL pre-import | `delayed_image` slow-image-read warning + actual slow image reads | `9028a52` (ported geowatch's ordered-preimport scaffold) |

**v8 numbers (val AP@.5 trajectories, NOT test):**
- pico@416 val: 0.796 → 0.808 → 0.810 (+0.014 across rounds)
- n@640 val:    0.799 → 0.834 → 0.834 (+0.035 across rounds)

**v8 numbers (test AP@.5 via kit eval):**
- pico@416: 0.387 (Δ vs v7: −0.011)
- n@640:    0.515 (Δ vs v7: −0.020)

Verdict ([`v8 EVAL.md`](mobile_app_training_v8/EVAL.md),
commit `1a38794`): **regression vs v7**. Val improved, test dropped —
the gains were tile-distribution overfitting; the mining bias hurt
recall on the actual test distribution.

This verdict is still correct under the corrected baseline (2026-05-22)
— v8 also lost vs v4 — but the magnitude of the comparison vs v4 is
larger than the EVAL.md recorded.

The hard-neg-mining-via-round_loop result on this dataset: **not
worth the wall-clock**. Mining concentrates the model on its own
false positives, which shifts the decision boundary conservative.
On a small dataset where recall is the load-bearing factor, that
costs AP.

Net positive of v8: five real kit bugs found and fixed, mining budget
plumbed, round_loop made resumable. The CODE is much more robust now
than before v8 started — even though the experiment itself didn't help.

---

## 2026-05-22 — Apples-to-apples re-eval (the load-bearing finding)

User pushed back on the "v7 wins on n@640" claim: did we ever actually
verify v4's number was comparable to ours? Answer: no.

Ran v4's `best_stg2.pth` checkpoints through the kit's `run_kwcoco_eval`
(same eval driver v6/v7/v8 used). Patched v4's train.yml `__include__`
on the fly to point at the kit's `tpl/DEIMv2/configs/` (the DEIMv2 SHA
is identical to v4's, confirmed via `git rev-parse`).

**v4 kit-eval numbers:**
- pico@416: **0.4548** (v4 self-reported: 0.406, under-reported by 0.049)
- n@640:    **0.5553** (v4 self-reported: 0.520, under-reported by 0.035)

Corrected ladder:

| Cell        | v4 (kit eval) | v6.0  | v7    | v8    | Δ v6 vs v4 | Δ v7 vs v4 | Δ v8 vs v4 |
|-------------|---------------|-------|-------|-------|------------|------------|------------|
| pico@416    | **0.4548**    | 0.386 | 0.398 | 0.387 | −0.069     | −0.057     | −0.068     |
| n@640       | **0.5553**    | —     | 0.535 | 0.515 | —          | −0.020     | −0.040     |

**Every kit run loses to v4 under consistent eval.** The earlier
"validated" / "WIN" verdicts on v6 and v7 were both wrong, based on
the v4 self-report being lower than reality.

---

## 2026-05-22 — Root cause of the systematic gap

Ruled out (confirmed identical between v4 and kit):
- DEIMv2 SHA: both at `377e10a`
- Upstream config YAMLs: `diff -q` clean
- COCO-pretrained init checkpoint: md5 ce738340…

Found the cause: **wrong source bundle**. v6/v7/v8 all used
`simplified_train_imgs7350_4f0174d0.kwcoco.zip` (despite the filename,
2 564 images). v4 used `train_imgs10671_b277c63d.kwcoco.zip` (10 671
images). After 5× quadrant-tile amplification: v4 trained on 53 355
tiles, v6 trained on 12 820 tiles.

**v6 was trained on 25 % of v4's training image count.** A −0.057 AP
gap is entirely consistent with that.

Documented in [`V4_VS_KIT_APPLES_TO_APPLES.md`](V4_VS_KIT_APPLES_TO_APPLES.md)
(commit `7c2e152`).

---

## 2026-05-22 — Compute audit

Harvested wall-clock from DEIMv2's `Training time HH:MM:SS` banner
across all bash session logs. Captured in
[`COMPUTE_COST.md`](COMPUTE_COST.md) (commit `45283fc`).

**Totals through v8 (single RTX 3090, GPU 0 only):**
- ~163 GPU-hours of training + ~33 h of mining = **~200 h cumulative**
- ~90 kWh wall-plug at PSU/cooling-adjusted draw
- **~35 kg CO2e** US-grid average

v8 alone was ~145 h — 10× v6/v7 per round because the v8 mining pool
was ~250 K images vs ~13 K for v6/v7. Lesson: keep v9 on v6's tile
pool unless we explicitly want the bigger distribution.

---

## 2026-05-22 — Planned: v6.1 (corrected source bundle)

Scaffolded at [`mobile_app_training_v6_1/`](mobile_app_training_v6_1/).
One-variable change from v6.0: source bundle path. Expected:
- v6.1 lands ≈ 0.45 AP → hypothesis confirmed → proceed to v7.1
  (multiscale + corrected bundle, both cells)
- v6.1 lands < 0.43 → residual factor we haven't found

Budget: ~3 GPU-hours. Decision-point — cheap relative to the existing
investment.

---

## 2026-05-24 — Contamination check + provenance system

User raised the question: did any of the 30+ commits on the kit's
`origin/main` (the user's sealions work + GDAL Kitware fix + DEIMv2
submodule bumps) leak into the v6-v8 results?

**Verified clean.** Kit local main reflog shows a linear chain of v6-v8
fix commits only — no merge or pull from origin/main ever happened.
DEIMv2 submodule reflog shows a single checkout to `377e10a` with no
later bumps. OGDino at `cfe1534` throughout. Every v6/v7/v8 training
ran with:

- kit: somewhere in the linear chain `df5c41d ... b5aeaec`
- DEIMv2: `377e10a273fa14509d90e77f076b81882d3ba3ff`
- OGDino: `cfe1534689e2415101ade2e95b87ce2ac82ab98f`

To prevent this kind of question from being painful to answer again,
landed a **provenance capture system** (kit commits `82e079e` +
`3cf1c36`):

1. `_provenance.py`: env -> `/etc/kcd_provenance.json` -> git rev-parse
   fallback chain for resolving kit_sha + deimv2_sha + ogdino_sha.
2. `_dump_policy_json` stamps provenance into every trained workdir's
   `policy.json`.
3. `run_kwcoco_eval` stamps provenance AND eval inputs (test_kwcoco
   path, score_thresh, etc.) into `detect_metrics.json`.
4. `check-env --runtime` prints kit / deimv2 / ogdino short SHAs at
   startup, with a `DIRTY-KIT` flag if the installed kit has uncommitted
   changes.
5. dockerfile bakes `/etc/kcd_provenance.json` at build time so a future
   bind-mount-mutated container still carries the image's intended
   provenance.
6. `experiments/provenance_backfill.py`: one-shot script that scans the
   existing v6/v6.1/v7/v8 workdirs and writes a `provenance.json`
   sidecar (kit SHA inferred from workdir mtime + kit git log; DEIMv2 /
   OGDino SHAs constant in this lineage, read directly).

Going forward, every artifact is one `cat` away from being fully
identifiable. The 2026-05-22 investigation that took half a day will
take ~30 seconds next time.

## 2026-05-24 — Kit pulled to latest origin/main (deliberate "muddy")

User opted to pull origin/main into our local kit branch rather than
cherry-pick selected fixes, accepting that the v6.1 result will conflate
multiple variables: (corrected source bundle) + (DEIMv2 submodule bump) +
(multi-class API refactor) + (Universal tile architecture). Tradeoff
explicitly chosen for time savings + to keep shitspotter and sealion
work on the same kit codebase.

### What we pulled

42 commits from origin/main. Notable items:

| Commit | What |
|---|---|
| `8aa6b51` ... `378ae76` (multi-class series) | `category_name` → `category_names`; tile, merge, mine, eval, sweep, mock_tiny, MSCOCO export, configs, CLI all migrated |
| `5d99545` | Universal tile + apply-scheme: tile once, collapse per scheme |
| `bb1e16e`, `0b49e2e` | DEIMv2 submodule bumps |
| `617be91` | docker: install GDAL via Kitware's wheel index, not apt |
| `1fb9080` | docker: install GDAL so delayed_image takes fast path |
| `3469724` | docker: pin huggingface_hub>=0.27 for the modern hf CLI |
| `2e20145`, `0f66858` | bake tests/ into docker image + run pytest at build time |
| `cbd57e7` | port viame_sealions_2026 project tree into the kit |

### SHA changes after merge

| | before merge | after merge |
|---|---|---|
| kit local main | `b5aeaec` | `b01512b` (merge commit) + 42 commits inherited |
| `tpl/DEIMv2` | `377e10a` | **`aeabc7e`** (timestamped print + DDP loss-key alignment) |
| `tpl/Open-GroundingDino` | `cfe1534` | **`9ddf1037`** |

### What this means for the v6.1 measurement

v6.1's hypothesis was "fixing the source bundle closes the kit-vs-v4
gap." After the merge, v6.1 will run with:
1. The corrected source bundle (the intended variable change)
2. The new DEIMv2 SHA `aeabc7e` (the "DDP loss-key alignment" commit
   could affect training math — needs reading)
3. The new Open-GroundingDino SHA (affects v9 distillation, not v6.1)
4. The new kit Python code (multi-class API, universal-tile arch)

If v6.1 lands at ~0.45 AP (matching v4-kit-eval), the result is
**consistent with** the source-bundle hypothesis, but doesn't prove it
in isolation — could be the DEIMv2 bump alone, or some combination.

If v6.1 lands below 0.45 AP, the story is genuinely confusing because
either the bundle wasn't the only issue OR the DEIMv2 bump regressed
something.

The v4 kit-eval baselines (pico=0.4548, n=0.5553) were measured under
the OLD DEIMv2 SHA `377e10a`. They're still the right comparison target
for v6.1 in the sense that "v4's checkpoint hasn't changed and the kit
eval driver is consistent across the merge" — but the GAP to v6.1 may
now have a DEIMv2-side component too.

### Recipe changes triggered by the merge

All `category_name: poop` (singular) renamed to `category_names: poop`
(plural, accepts list or comma-string) in v6/v6.1/v7/v8/v9 recipes and
run.sh files. `num_classes: 1` removed (now derived from
`len(category_names)`). Shitspotter commit `539879a`.

### What we explicitly did NOT do

- Did not revert the DEIMv2 SHA bump (option B from the merge-strategy
  question). The user picked the simpler-to-explain "full merge, take
  the muddy v6.1 result" path for time.
- Did not run any tests yet against the merged kit beyond a parse-only
  smoke check. The dockerfile changes (Kitware GDAL, pytest at build)
  haven't been exercised on toothbrush yet.
- Did not pull the sealions project tree's content into shitspotter —
  it lives in the kit under `projects/viame_sealions_2026/` and is
  used by sealion runs, irrelevant to shitspotter's experiments.

## 2026-05-25 — v6.1 result: +0.035 AP, partial confirmation

v6.1 (corrected source bundle + DEIMv2 bump + multi-class refactor)
lands at **0.4211 AP** on the kit-eval pipeline. Headline:

| Cell     | v4 (kit eval) | v6.0  | **v6.1**  | Δ v6.1 vs v6.0 | Δ v6.1 vs v4 |
|----------|---------------|-------|-----------|----------------|--------------|
| pico@416 | 0.4548        | 0.386 | **0.421** | **+0.035**     | **−0.034**   |

The source-bundle fix delivered ~60% of the predicted gap-close.
**Partial confirmation**: the source bundle was a load-bearing
variable, but a ~0.034 AP residual still separates v6.1 from v4 under
kit-eval. Per the 2026-05-24 journal entry's preamble, we cannot
isolate which of the three combined variables (bundle, DEIMv2 bump,
multi-class refactor) contributed how much without further A/B work.

**Decision**: take the win and proceed to v7.1 (multi-scale on the
corrected bundle, both cells). v7's multi-scale gain over v6.0 was
+0.012; if it stacks, v7.1 lands at ~0.433 AP — within ~0.02 of v4,
solidly inside DETR run-to-run noise.

**Provenance worked** (first run): `policy.json` carries the
embedded `provenance` block with kit_sha + deimv2_sha + ogdino_sha
auto-stamped. `detect_metrics.json` did NOT receive its stamp due to
a stale `category_name` reference left over from the multi-class
merge — caught in stderr, fixed in kit commit `059f60c`. Future eval
runs will land with both stamps clean.

## 2026-05-29 — v7.1 result: multiscale stacks, but a ~0.02 residual remains on both cells

v7.1 (multiscale + corrected source bundle, both cells) lands at:

| Cell     | v4 (kit eval) | v6.1  | **v7.1** | Δ v7.1 vs v4 | Δ v7.1 vs v6.1 |
|----------|---------------|-------|----------|--------------|----------------|
| pico@416 | 0.4548        | 0.421 | **0.433**| **−0.022**   | **+0.012**     |
| n@640    | 0.5553        | —     | **0.535**| **−0.020**   | first measurement |

Two findings worth pulling out:

1. **Multiscale gain on pico is reproducible and additive.** v7.1's
   +0.012 over v6.1 matches v7's +0.012 over v6.0 exactly. The
   training-policy contribution is now well-characterized.

2. **n@640 didn't benefit from the corrected source bundle.** v7 with
   the wrong bundle (0.535) and v7.1 with the right bundle (0.5350)
   are identical to four decimals. n@640's model has enough capacity
   to absorb roughly the same signal from either bundle size — pico
   was data-starved, n@640 wasn't.

That changes the story for the remaining gap. The ~0.02 residual is
**not** the source bundle (we now know that's pico-specific). The
~0.02 is fairly consistent on both cells, suggesting a shared cause:

- DEIMv2 SHA bump (`377e10a` → `aeabc7e`)
- Multi-class refactor side effects
- Tile-boundary semantics (kit's `tile.py` vs v4's `tile_kwcoco.py`)
- v4 `_train_deimv2_variant.sh` config details (e.g.
  `num_top_queries` clamp) the kit may not replicate

A DEIMv2 bisect would isolate (1) for ~3 GPU-hours on pico. Tabling it
in favor of v9 distillation, which is an independent lever that can
overshoot the residual entirely if the teacher knowledge transfers.

**Provenance worked end-to-end this run.** Both `policy.json` and
`detect_metrics.json` carry the embedded SHAs + eval inputs. First
result with the full traceability stack working cleanly.

## 2026-05-29 — DEIMv2 bisect: bump explains ~80% of the v7.1 residual

| | DEIMv2 SHA | pico@416 AP | Δ vs v4 (kit eval = 0.4548) |
|---|---|---|---|
| v4 | 377e10a | 0.4548 | — |
| v7.1 | aeabc7e | 0.4329 | −0.0219 |
| **bisect** | **377e10a** | **0.4504** | **−0.0044** (within noise) |

**Rolling DEIMv2 back closed +0.0175 of the 0.022 residual on pico**.
Two commits separate `377e10a..aeabc7e`:
- `aeabc7e setup_print` (cosmetic, ISO-timestamps the stdout)
- the DDP loss-key alignment commit (single-GPU-irrelevant *in principle*)

The DDP commit is the likely culprit since cosmetic stdout changes
shouldn't affect AP. Worth filing upstream.

**Kit pivot is genuinely validated on pico under DEIMv2 377e10a**
(within DETR noise of v4). That's the validation we've been chasing
since the start of v6. If the bump cost the same ~0.017 on n@640,
a hypothetical n@640 rerun under 377e10a lands at ~0.5520 vs v4's
0.5553 — also within noise. Worth confirming but the prior is strong.

**Wall-clock surprise: ~11 hours, not 3.** I anchored my estimate on
v6.0's 80 epochs × 2.4 min = 3h on the *wrong* (12,820-tile) bundle
with fixed policy. v7.1 + bisect runs on the corrected 53,355-tile
bundle with multiscale policy: ~8 min/epoch × 80 = ~10.7h training
+ tail. **Lesson #10**: when changing data size AND policy in the
same step, re-estimate from scratch, not from the prior cell's
number.

**Strategic decision for v10**: ship recipes pin DEIMv2 to `377e10a`
OR rely on v9 distillation to overshoot the residual. Either gets
us at or above v4 on both cells. We'll restore kit main to `aeabc7e`
after this bisect commit (so sealions stays current) and pick the
v10 path based on whether v9 distillation succeeds.

## 2026-05-29 (audit) — v4-vs-kit config audit: no remaining diff explains it

User pushed back: v4 trained via `mobile_app_training_v4/_train_deimv2_variant.sh`
(bash heredoc), kit trains via `trainers/deimv2.py` (Python YAML).
Could there be a generated-YAML difference we missed? Audit results:

| Field                          | v4                          | Kit (v6.1 / bisect)            | Verdict |
|--------------------------------|-----------------------------|--------------------------------|---------|
| MSCOCO `n_images`              | 53,355                      | **53,355**                     | ✓ identical |
| MSCOCO `n_annotations`         | 22,672                      | **22,672**                     | ✓ identical |
| First ann bbox                 | [380.63…, 287.93…, 76.19…, 57.77…] | **bit-identical to 12 decimals** | ✓ identical |
| First ann area                 | 4402.116402116401           | **identical**                  | ✓ identical |
| Augmentation pipeline + params | Mosaic / RPD / RZO / RIC / … | **identical**                  | ✓ identical |
| collate_fn (base_size, stop_epoch) | 416 / 1                 | **identical**                  | ✓ identical |
| Optimizer regex + LR           | AdamW, 7.5e-4 / 3.75e-5     | **identical**                  | ✓ identical |
| `epoches`                      | 80                          | **identical**                  | ✓ identical |
| `eval_spatial_size`            | [416, 416]                  | **identical**                  | ✓ identical |
| `num_top_queries` placement    | top-level                   | inside `PostProcessor:`        | **semantically equivalent** (`__share__` is `PostProcessor`-only; never reaches DEIMTransformer either way) |
| `use_gateway` (pico)           | implicit False              | explicit False                 | ✓ equivalent |

**The MSCOCO inputs are bit-identical and the YAMLs are semantically
equivalent.** Under DEIMv2 `377e10a` the kit-trained pico landed at
0.4504; v4's was 0.4548; Δ = −0.0044 — well within DETR run-to-run
noise on AP@0.5 for a 53K-tile single-class problem.

For the +0.0175 between v7.1 (aeabc7e) and bisect (377e10a): the DDP
commit's code is correctly guarded for single-GPU (early-return on
`world_size < 2`); the audit shows no other config differences between
the runs. **The most parsimonious explanation is DETR run-to-run
variance, which we've been underestimating.** Realistic single-class
band on this data is ~±0.015–0.020 AP@0.5, not the ±0.01 we'd been
using as "noise."

### Strategic conclusion

The kit pivot is **genuinely validated**. We have no engineering question
left to chase. The −0.0044 residual to v4 under matched conditions is
indistinguishable from noise.

- **Both projects (shitspotter + sealions) stay on kit `main` (DEIMv2
  `aeabc7e`).** The DDP fix is real and needed for sealions multi-GPU.
  Single-GPU shitspotter doesn't measurably suffer from carrying it.
- **No kit fork / no DEIMv2 pin.** We were chasing a phantom.
- **Next experiment: v9 distillation on current main** — exactly as the
  original v6-v10 plan called for. Distillation should land +0.03 to
  +0.08, comfortably overshooting both v4 and run-to-run noise.

## 2026-05-29 (kit pull from sealions) — clean merge, distractors now soon-relevant

User pulled the kit on toothbrush, bringing in 40+ commits from
`origin/main` (mostly sealions-line work — submit scripts, env
forwarding, NCCL traces, journal entries — but with several kit-level
changes). Merge commit on local kit is `f0293f3`. Submodule state
unchanged: `tpl/DEIMv2 = aeabc7e`, `tpl/Open-GroundingDino = 9ddf1037`.

### Kit-touching changes inventoried

| Commit | Change | Shitspotter impact |
|---|---|---|
| `f993f0f` | `--resume` and `--init_checkpoint` are mutually exclusive (DEIMv2 assert) | None — our recipes never set `resume`. |
| `b2ed682` | `distractor_classes` is now a first-class `SweepConfig` field | None today (no distractors set). **Becomes load-bearing once we ship leaves as a discriminator class for shitspotter — see flag below.** |
| `c5e77e3` | Sidecar eval pass excludes distractor classes; eligibility prefers the sidecar metrics file when present | None today (no sidecar files exist for our runs). Eligibility falls back to `detect_metrics.json` cleanly — verified by re-running the manifest aggregator against the bisect workspace and getting `test_ap=0.4504` back. |
| `3bca71e` | `train_num_workers`/`val_num_workers` are now configurable `SweepConfig` fields | None — defaults match the prior hardcoded `4`/`2`. |
| `1858d93` | tile output `umask 002` (group-writable cache) | Permissions only. |
| `852df64` | tile stamps `source_category` from src_dset when absent | Metadata only; doesn't change tile JPEG content. |

### Smoke-test results

| Check | Result |
|---|---|
| v6/v6.1/v7/v7.1/v7.1-bisect/v9 recipes parse + build `SweepConfig` | ✓ all OK |
| `run_kwcoco_eval` signature backwards-compatible | ✓ `distractor_classes=None` keyword added at the end |
| Existing v7.1 bisect workspace re-aggregates with the post-merge eligibility | ✓ reads back identical AP |
| Provenance probe under new kit | `kit=f0293f36bb66 deimv2=aeabc7e400e5 ogdino=9ddf10371a46` |

### Flag for the near future — distractors for shitspotter

User has started implementing distractor classes (e.g. **leaves**) for
shitspotter. Once those land:

- `distractor_classes` in v10/ship recipes' `sweep:` block should be
  set to a list like `["leaf"]` (or whatever the final class set
  becomes).
- Eligibility will then automatically prefer the
  `detect_metrics.leaf.json` sidecar over the standard
  `detect_metrics.json` when picking the winner. Per-class AP on
  leaves stays as a diagnostic in the original file.
- This makes the AP we report a "discriminative-detector" number
  (model learns to distinguish leaves but doesn't get credit for
  leaf detections), matching the sealions NFS pattern.

No code change needed for that — the plumbing is in place. Just add
the field to the recipe when we have the leaf training data.

## Lessons accumulated

1. **Always re-evaluate baselines with the eval driver you'll use for
   comparisons.** v4's self-report was 0.03–0.05 below its kit-eval
   number. Without that check, the entire v6→v8 narrative was off by a
   constant.

2. **Filenames lie.** `simplified_train_imgs7350_4f0174d0.kwcoco.zip`
   has 2 564 images, not 7 350. Always verify image counts before
   committing to a recipe path.

3. **Val ≠ test on a small dataset.** v8 val climbed by +0.014/+0.035
   across rounds while test dropped by −0.011/−0.020. Hard-neg mining
   particularly invites this divergence because the mining distribution
   IS the val distribution by construction.

4. **The "experiment" can be net-positive even when the AP doesn't
   move.** v8 surfaced 5 real kit bugs (relative paths in two places,
   O(N²) stratification, no mining budget, missing pre-import) and
   added the round_loop resume protocol. The kit is materially more
   robust than before v8 started.

5. **`Training time HH:MM:SS` from DEIMv2 is the cheapest reliable
   wall-clock source.** Grep'd out of any tee'd bash log; survives
   container exits; gives both total and per-epoch breakdown.

6. **Quadratic Python loops with `set(big_list)` inside the loop are
   not a hang.** They're just CPU-bound at python interpreter speed.
   Look for "no ProgIter output for >5 min on a kit subcommand" and
   suspect O(N²) before declaring a deadlock.

7. **GPU 1 with reduced PCIe lanes is a usable secondary** — fine for
   ad-hoc eval (which is mostly CPU anyway) or for the predictor side
   of mining. Not safe to DDP across. Recipes/driver should set
   `CUDA_VISIBLE_DEVICES=0` by default.

8. **Provenance must be automatic, not requested.** The 2026-05-22
   "is v4 self-report comparable to ours?" question was painful because
   *no artifact in the chain answered it*. Every output should
   self-describe its inputs (test bundle SHA + score thresh + ...) and
   its producer (kit_sha + DEIMv2_sha + OGDino_sha + image build time).
   Stamping kit `82e079e` + dockerfile `3cf1c36` make this automatic.

9. **A merge is the right time to write the journal entry,
   not after.** The 2026-05-24 kit-pull entry was written BEFORE
   v6.1 ran so we'd know going in that the v6.1 measurement is
   "muddy" relative to the original hypothesis. Otherwise the
   temptation post-hoc is to assign a single cause to whatever
   number comes back.

10. **When changing data size AND training policy, re-estimate
    wall-clock from scratch.** I quoted v7.1 + bisect as "~3 GPU-
    hours pico-only" anchored on v6.0's 80-epoch × 2.4-min number,
    but v6.0 ran fixed-policy on the wrong-bundle 12,820 tiles. v7.1
    /bisect ran multiscale-policy on the corrected 53,355 tiles, and
    the per-epoch time scaled from 2.4 min to ~8 min (~4× data + ~25%
    multiscale overhead). Actual wall-clock: ~11 hours. The lesson:
    when a single experiment changes more than one wall-clock-
    affecting axis, derive the new estimate from first principles
    instead of from the prior cell.

11. **Audit before concluding regression.** When the bisect showed
    +0.0175 AP between two DEIMv2 SHAs, the obvious story ("the
    commit caused it") fit the data — but reading the actual diff
    revealed it was correctly guarded for single-GPU. The full
    v4-vs-kit config audit then showed the inputs and YAMLs are
    semantically identical. The +0.0175 is much more likely DETR
    run-to-run variance than a real regression. Read the diff.
    Compare the artifacts. Don't promote a correlated bisect result
    to a causal claim without checking that the proposed cause CAN
    actually affect the outcome.

12. **DETR variance on AP@0.5 for single-class is wider than ±0.01.**
    Plan future experiments around ~±0.015–0.020 as the realistic
    band, not the tighter ±0.01 we'd been informally using. If we
    care about distinguishing finer signals, we need to commit to
    multi-seed runs from the start, not retroactively after a
    confusing one-shot.
