# ShitSpotter Repository Guidance

- **Project overview**: Computer-vision tooling for the ShitSpotter / ScatSpotter dataset and models (see `README.rst`). The package hosts data acquisition utilities, dataset split management, training / inference entrypoints, and analytic scripts.
- **Key references**:
  - `README.rst` – project goals, data distribution, high-level milestones.
  - `dev/journal.txt` – running development log, troubleshooting notes, experiment context.
  - `papers/` – in-progress manuscripts; respect their structure when editing.
  - `AGENTS.details.md` – extended 1–2 page orientation covering core components, workflows, and operational tips.
- **Directory orientation**:
  - `shitspotter/` – main Python package (data tooling, training, prediction, analysis helpers).
  - `experiments/` – experiment records and repro steps; scripts may have bitrot but aim for end-to-end reproducibility.
  - `dev/` – prototypes, works-in-progress, and miscellaneous development
    byproducts; treat contents as references rather than production utilities.
- **Testing & linting**: Prefer `python run_tests.py` (pytest + coverage + xdoctest). Supplement with `./run_doctests.sh` / `./run_linter.sh` as appropriate.
- **Data & models**: Distribution endpoints (IPFS, HuggingFace, torrents) are
  coordinated through helpers in `shitspotter/ipfs.py` and
  `shitspotter/phone_manager.py`. Experiment folders record which assets tie to
  which paper results.
- **General practices**:
  - Keep docstrings / inline notes informative—many files double as documentation.
  - Favor clarity and reproducibility; document non-obvious steps, data paths, and experiment parameters.
  - Be mindful of large external datasets / models; avoid hardcoding environment-specific secrets or paths.
