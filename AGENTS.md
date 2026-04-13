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
  - For long-running experiment scripts, prefer atomic final artifacts: either emit the canonical "final" outputs only after the stage completes successfully, or write explicit run-state metadata alongside intermediate outputs so resumable scripts can distinguish complete artifacts from partial/failed ones.
  - Be mindful of large external datasets / models; avoid hardcoding environment-specific secrets or paths.


## Behavior
* Think like a computer vision researcher, when you see a new result it is
  important to record it in our research journal

## Developer journal
Keep a running journal at `dev/journals/<agent_name>.md` (e.g.
`dev/journals/codex.md`) to capture the story of the work (decisions, progress,
challenges). This is not a changelog.  Write at a high level for future
maintainers: enough context for someone to pick up where you left off.

- Format: Each entry starts with `## YYYY-MM-DD HH:MM:SS -ZZZZ` (local time).
- Must include: summary of user intent. If still editing this entry then you can update this if the user provides a clarifying prompt. The idea is that this is a compressed form of the prompt you are responding to. You can refer to previous entries to keep this concise.
- Must include: your exact model name and configuration.
- Must include: what you were working on, a substantive entry about your state of mind / reflections, uncertainties/risks, tradeoffs, what might break, what you're confident about.
- May include: what happened, rationale, testing notes, next steps, open questions.
- Rules: Prefer append-only. You may edit only the most recent entry *during the same session* (use timestamp + context to judge); never modify the timestamp line; once a new session starts, create a new entry. Never modify older entries. Avoid large diffs; reference files/modules/issues instead.
- Write journal entries as design narratives: capture the user's underlying goal, the constraints that matter, the alternatives you considered, why the chosen approach won, what tradeoffs were accepted, and 1-3 reusable design takeaways that could teach a future engineer how to make a similar decision.

# Agent Guidelines for ShitSpotter

## Always end with rerun commands

After any change to scripts, evaluations, or inference pipeline code, always
finish your response with the concrete shell commands the user should run to
exercise the new code.  This lets them copy-paste immediately without having to
reconstruct the invocation themselves.