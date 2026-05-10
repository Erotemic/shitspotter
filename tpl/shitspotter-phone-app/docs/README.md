# `tpl/shitspotter-phone-app/docs/` — read order

Six docs, in the order a brand-new agent should read them:

1. **[000_stack_decision.md](000_stack_decision.md)** —
   Why KMP + Compose Multiplatform was chosen, what was rejected,
   what trade-offs are non-obvious. The strategic baseline.

2. **[001_build_run_validate.md](001_build_run_validate.md)** —
   Build + sideload commands; toolchain sanity check; how to point
   the runtime at a real ONNX model.

3. **[002_benchmarks_template.md](002_benchmarks_template.md)** —
   The schema future benchmark reports must follow per GOAL.md
   "Performance discipline".

4. **[003_known_limitations.md](003_known_limitations.md)** —
   17 explicit follow-ups grouped by hot-path / build / coverage /
   harness / capture / docs / low-pri. Pick from here when adding
   the next agent's PR.

5. **[004_kotlin_python_parity.md](004_kotlin_python_parity.md)** —
   Milestone-2 reference comparison: Kotlin vs Python on the same
   YOLOX-nano model + dog.jpg input.

6. **[005_runtime_architecture.md](005_runtime_architecture.md)** —
   Per-package responsibilities, hot-path buffer count, lifecycle
   table. The deepest "where does this code live?" map.

## Plus

- `benchmarks/` — concrete report archive. `2026-05-10_desktop_dog_jpg_baseline.md`
  is the parity baseline; future Pixel-5 reports go beside it.
- `benchmarks/.keep` — placeholder so an empty-but-existing directory
  survives `git clean`.

## Cross-references that aren't in this folder

Outside `docs/` but worth knowing:

- [`../GOAL.md`](../GOAL.md) — the original goal document; lives at the
  in-tree app folder root. `../AGENT_GOAL.md` is a symlink to it.
- [`../README.md`](../README.md) — short-term breadcrumbs + UI wireframe.
- [`../scripts/run_all_desktop_validation.sh`](../scripts/run_all_desktop_validation.sh) —
  the one-shot validation pass; running this is the first sanity
  check after any non-trivial change.
- [`../../../dev/journals/2026-05-10_phone_app_kmp_scaffold.md`](../../../dev/journals/2026-05-10_phone_app_kmp_scaffold.md) —
  the run journal for the original scaffold.
