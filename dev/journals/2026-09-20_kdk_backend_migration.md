# KDK backend ownership migration — 2026-09-20

## Decision

Generic detector / segmenter source repositories belong to
`kwcoco_detector_kit` (KDK), not to ShitSpotter. The long-term goal is for KDK
to support every detector family used seriously in historical ShitSpotter work
through one KWCoco data contract, trainer API, prediction format, evaluator,
and provenance record. Historical techniques are therefore retained as tested
backends rather than deleted as obsolete experiments.

ShitSpotter continues to own poop-specific experiment policy, data selection,
learned model artifacts, and application code.

## Repositories moved to KDK ownership

The following ShitSpotter submodules are removed as independently-owned
submodules and are canonical under `kwcoco_detector_kit/tpl/`:

- `DEIMv2`
- `Open-GroundingDino`
- `segment-anything-2`
- `MaskDINO`
- `YOLOX`
- `YOLO-v9`

ShitSpotter retains these domain-specific submodules:

- `tpl/poopdetector`
- `tpl/poop_models`
- `tpl/scatspotter_app`

For compatibility with historical scripts that hard-code the old
`shitspotter/tpl/...` paths, the removed generic gitlinks are replaced by thin
relative symlinks into the sibling KDK checkout. New shared experiment setup
uses `experiments/backend_repos.sh` and resolves KDK directly instead of relying
on those compatibility links.

The default checkout layout is:

```text
~/code/
    shitspotter/
    kwcoco_detector_kit/
```

Override `KWCOCO_DETECTOR_KIT_DPATH` when that layout is not used. The tracked
compatibility symlinks specifically assume the sibling layout and are intended
as a transition aid until the historical shell experiments are represented as
first-class KDK recipes.

## Historical source pins captured before migration

These are the exact ShitSpotter submodule commits observed immediately before
removing the duplicate gitlinks:

| Backend | Historical ShitSpotter commit | Historical branch state |
| --- | --- | --- |
| DEIMv2 | `377e10a273fa14509d90e77f076b81882d3ba3ff` | `shitspotter` |
| Open-GroundingDino | `b59dd5e7a4c66a5481c80e24196f2a9e728b380b` | detached HEAD |
| segment-anything-2 | `6868c98bab05f521aea9c77aebcc6963990a6653` | `shitspotter` |
| MaskDINO | `c4a9d500a838b0a2e6f9de6061b6475a708b3243` | `shitspotter` |
| YOLOX | `586829f1bf6878fa8953f23ac9e3d5df9ed81642` | `shitspotter` |
| YOLO-v9 | `543599d8d5dfc6f9035c775e21171a04d1b27d41` | `shitspotter` |

The KDK checkout immediately after adding the missing historical repositories
was:

| Backend | KDK commit at migration |
| --- | --- |
| DEIMv2 | `1e6339dcf351f63a6c119d41363d55b9a34124fa` |
| Open-GroundingDino | `9ddf10371a46ddca080b9319306185bc704325e5` |
| segment-anything-2 | `6868c98bab05f521aea9c77aebcc6963990a6653` |
| MaskDINO | `c4a9d500a838b0a2e6f9de6061b6475a708b3243` |
| YOLOX | `586829f1bf6878fa8953f23ac9e3d5df9ed81642` |
| YOLO-v9 | `543599d8d5dfc6f9035c775e21171a04d1b27d41` |

The last four were exact ownership moves: ShitSpotter and KDK pointed at the
same commits. DEIMv2 and Open-GroundingDino deliberately differ because KDK
already contains newer/fixed integration commits.

**Do not treat KDK's current DEIMv2 or Open-GroundingDino checkout as an exact
reproduction of the old ShitSpotter experiment merely because the old path now
resolves through KDK.** Exact historical KDK recipes must record and verify the
historical source commit above, or explicitly document why a modernized backend
commit is being used.

## Architectural target

Each historical backend should eventually have three pieces in KDK:

```text
tpl/<upstream-or-fork>/
kwcoco_detector_kit/trainers/<backend>.py
tests/backends/<backend>/
```

The stable API is the KDK adapter, never the `tpl/` repository itself. A backend
adapter should normalize at least:

1. KWCoco -> backend-native data/config conversion.
2. Variant and historical-recipe selection.
3. Training launch and distributed launch semantics.
4. Checkpoint discovery and resume behavior.
5. Prediction -> KWCoco conversion.
6. Common evaluation.
7. Export/deployment capabilities where supported.
8. Exact provenance: KDK commit, upstream/fork commit, generated config, input
   manifest identities, environment, and recipe identifier.
9. CPU-safe config/conversion tests plus optional GPU smoke tests.

Backend capabilities should be declared rather than faked. For example YOLOX
is bbox-only while MaskDINO and RF-DETR Seg have native instance masks, and
SAM2 is primarily a segmentation stage rather than a standalone detector.

## Migration follow-up

- DEIMv2 and Open-GroundingDINO already have first-class KDK trainer plugins.
- SAM2 already has substantial KDK integration; its upstream checkout is now
  colocated with that code.
- Add real KDK trainer/predictor adapters and tests for MaskDINO, YOLOX, and
  YOLO-v9 instead of preserving independent ShitSpotter training harnesses.
- Convert historically important ShitSpotter recipes into named KDK recipes
  with their original source pins and hyperparameters.
- Once those recipes exist, remove the compatibility `tpl/...` symlinks and
  change remaining archival scripts/docs to point at KDK recipe identifiers
  rather than raw source-tree paths.
