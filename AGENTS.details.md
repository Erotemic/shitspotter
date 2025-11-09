# ShitSpotter Orientation (Extended)

This document complements `AGENTS.md` with a broader tour of the project so a new
contributor can find the major systems quickly. Treat it as a searchable map:
links and filenames are fully qualified to simplify navigation.

## Project goals, data, and documentation

* **Mission** – Build and deploy models that can locate pet waste in camera
  phone images images and package as a phone app. The repo stores code
  for data acquisition, dataset management, training, evaluation, and packaging.
* **Read next** – `README.rst` covers project history, dataset distribution, and
  milestone tracking. `DATASHEET.md` summarizes collection ethics and usage
  considerations. The living `dev/journal.txt` log captures experiment notes and
  architectural decisions.
* **External docs** – Sphinx documentation is under `docs/` and published on
  ReadTheDocs. Dataset/model hosting is coordinated through IPFS, HuggingFace,
  and torrents; scripts in `shitspotter/ipfs.py` and
  `shitspotter/phone_manager.py` show how assets are pinned, mirrored, and named.

## Repository layout (quick reference)

* `shitspotter/` – Installable Python package.
  * **Data ingestion** – `gather.py` and `gather_from_staging.py` import raw
    phone captures, scrub EXIF metadata, and map files into the kwcoco dataset
    structure. `snapshot_dataset.py` and `transmission.py` publish curated
    bundles to distribution endpoints.
  * **Dataset management** – `make_splits.py` constructs train/val/test partitions
    for kwcoco manifests. `util/util_data.py` centralizes path discovery for DVC
    roots, staging folders, and manifest files. CLI helpers such as
    `cli/simplify_kwcoco.py` and `cli/extract_polygons.py` manipulate annotations
    without editing the raw data by hand.
  * **Training + inference** – `detectron2/fit.py` implements configurable
    Detectron2 training loops. `detectron2/predict.py` and `pipelines.py`
    support batched inference, post-processing, and export flows. `matching.py`
    contains data association logic for before/after/negative image triplets.
  * **CLI entrypoints** – Modules under `shitspotter/cli/` are exposed through
    console scripts declared in `setup.py`. They handle prediction
    (`cli/predict.py`), metadata scrubbing (`cli/scrub_exif.py`), statistics
    (`cli/coco_annotation_stats.py`), and dataset transformations.
  * **Model variants** – `other/` holds pipelines and predictors for alternative
    architectures (YOLO, GroundingDINO, etc.) used in exploratory experiments.
  * **Analysis** – `plots.py` contains figure-generation utilities used by
    notebooks and experiment summaries.
* `experiments/` – Hands-on experiment logs. Each directory typically includes a
  `README.md`, dockerfile pointers, and reproduction scripts. Expect occasional
  bit-rot; comments usually note required dataset revisions or external commits.
* `dev/` – A grab bag of works-in-progress, scratch notebooks, prototypes, and
  other development byproducts. Expect uneven polish; treat findings here as
  references rather than ready-to-ship tooling.
* `papers/` – Draft manuscripts, submission artifacts, and measurement logs.
* `docs/` – Sphinx project for generated documentation (`make html` builds the
  site locally).
* Automation – `run_tests.py`, `run_doctests.sh`, `run_linter.sh`, and
  `run_developer_setup.sh` standardize local checks and environment bootstraps.

## Data lifecycle and storage

1. **Capture & staging** – Phone uploads land in a staging directory.
   `shitspotter/gather_from_staging.py` formalizes the move into the repository's
   managed structure and can remove sensitive EXIF metadata via
   `shitspotter/cli/scrub_exif.py`.
2. **Ingestion & manifests** – `gather.py` assembles before/after/negative
   triples, reconciles annotations, and outputs kwcoco manifests. Supporting
   utilities in `util/util_data.py` locate canonical paths for these assets.
3. **Versioning** – Releases are content-addressed. `ipfs.py` wraps pin/publish
   operations, while `phone_manager.py` and `transmission.py` coordinate uploads
   to IPFS, Girder mirrors, torrents, and HuggingFace.
4. **Tracking revisions** – `cid_revisions.txt` and `pin_table.txt` list known
   dataset CIDs. Update these files and the `dev/journal.txt` log when publishing
   new snapshots or models.

## Model development workflow

* **Configuration** – Training presets live in `detectron2/fit.py` and template
  YAML snippets under `tpl/`. Align dataset names with manifests produced by
  `make_splits.py`.
* **Training** – Run `python -m shitspotter.detectron2.fit` or use `train.sh` for
  convenience wrappers that set environment variables and log directories.
* **Evaluation** – Use `python -m shitspotter.detectron2.predict` or
  `shitspotter/cli/predict.py` to generate detections. Downstream quality checks
  rely on helpers in `plots.py` and experiment notebooks.
* **Exporting for deployment** – `pipelines.py` and `phone_manager.py` provide
  hooks for quantization, ONNX conversion, and bundling artifacts for mobile
  prototyping or IPFS distribution.

## Testing, linting, and CI

* `python run_tests.py` runs pytest with coverage and xdoctest integration.
* `./run_doctests.sh` focuses on docstring examples; `./run_linter.sh` wraps the
  repo's formatting and static-analysis defaults.
* CI configurations in `.github/workflows/` mirror these commands. Match the
  workflows locally when debugging failures.

## Environment and dependencies

* `setup.py` currently declares `python_requires ">=3.7"`, but most production
  scripts assume modern Python (3.9+) with CUDA-enabled Detectron2 for training.
* Dependency pins live in `requirements.txt` and the `requirements/` directory.
  kwcoco and ubelt are typically safe to bump, but Detectron2 (Mask R-CNN
  builds), YOLOX integrations, and GroundingDINO experiments have fragile
  version constraints.
* Several dockerfiles exist under `dockerfiles/` to approximate known-good
  environments. They help document setup steps but have not been validated end
  to end recently—expect to troubleshoot GPU drivers and private data mounts.
* Encrypted and rotatable secrets and credentials for publishing assets live
  under `secrets/`; never commit modifications there. Scripts read environment
  variables documented in their module docstrings.

## Operational tips

* Many commands expect large external datasets that live outside the repo. Check
  docstrings for path assumptions (often `ub.Path.appdir` caches) before running
  scripts on a fresh machine.
* Document non-obvious behavior in `dev/journal.txt` so others can trace the
  rationale behind new experiments or data pushes.
* Reuse kwcoco utilities and CLI tools instead of writing ad-hoc data munging
  code; this keeps manifests consistent across experiments.
* When publishing new assets, verify IPFS pins with `ipfs.py` helpers and record
  the resulting CID alongside reproduction notes in `experiments/`.

## Where to dive deeper

* Search for `TODO` / `FIXME` markers to find active work items.
* `experiments/` contains reference pipelines for results reported in `papers/`.
* The combination of `README.rst`, `AGENTS.details.md`, and recent entries in
  `dev/journal.txt` should give enough context for onboarding and planning.
