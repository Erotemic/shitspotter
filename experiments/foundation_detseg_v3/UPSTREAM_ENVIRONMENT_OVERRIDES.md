# Foundation Det+Seg V3 Upstream Environment Overrides

This file records the intentional ways the ShitSpotter foundation-detseg setup
differs from upstream DEIMv2 / SAM2 / MaskDINO install instructions so the
local developer environment stays coherent.

## Current overrides

### DEIMv2

- Upstream file:
  [tpl/DEIMv2/requirements.txt](/home/agent/code/shitspotter/tpl/DEIMv2/requirements.txt)
- Upstream behavior:
  pins `torch==2.5.1` and `torchvision==0.20.1`
- ShitSpotter override:
  [setup_environment.sh](/home/agent/code/shitspotter/experiments/foundation_detseg_v3/setup_environment.sh)
  filters out those two pins and preserves the existing torch stack
- Why:
  the upstream README says `2.5.1` and `2.4.1` are the tested versions and
  recommends `2.0+`, and core DEIMv2 imports succeeded here on Torch `2.10.0`
  and Torchvision `0.25.0`
- Risk:
  newer Torch/Torchvision combinations are not formally upstream-tested, so do a
  quick GPU sanity run before launching a long training job
- Additional local patch:
  [tpl/DEIMv2/engine/data/transforms/_transforms.py](/home/agent/code/shitspotter/tpl/DEIMv2/engine/data/transforms/_transforms.py)
  was patched to implement `transform(...)` instead of the older `_transform(...)`
  hook for three custom torchvision-v2 transforms (`PadToSize`,
  `ConvertBoxes`, and `ConvertPILImage`)
- Why:
  `torchvision 0.25.0` dispatches custom `T.Transform` subclasses through
  `transform(...)`, and the upstream DEIMv2 code was still targeting the older
  hook, which caused a bare `NotImplementedError` during the first training
  batch
- Risk:
  this is a local compatibility patch against upstream DEIMv2 and may need to
  be revisited if the submodule is updated or if upstream merges their own fix

### MaskDINO

- Upstream file:
  [tpl/MaskDINO/requirements.txt](/home/agent/code/shitspotter/tpl/MaskDINO/requirements.txt)
- Upstream behavior:
  installs `opencv-python`
- ShitSpotter override:
  [setup_environment.sh](/home/agent/code/shitspotter/experiments/foundation_detseg_v3/setup_environment.sh)
  filters out `opencv-python`, preserves any existing `cv2` provider, and only
  installs `opencv-python-headless` if `cv2` is missing entirely
- Why:
  the GUI OpenCV wheel conflicts with headless-first environments, and MaskDINO
  documents OpenCV as optional for demo / visualization rather than a core model
  requirement in [tpl/MaskDINO/INSTALL.md](/home/agent/code/shitspotter/tpl/MaskDINO/INSTALL.md)
- Risk:
  interactive OpenCV-window demos may still prefer the non-headless wheel

## Guidance

- Append new overrides here when local setup diverges from upstream on purpose.
- Keep this as rationale and policy, not as a command log.
- If an override is removed later, leave a short note rather than rewriting
  history heavily.
