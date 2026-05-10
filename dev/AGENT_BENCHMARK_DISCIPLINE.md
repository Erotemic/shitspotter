# ShitSpotter Agent Discipline: Hard-Problem Benchmark Memory

This file is for agents working in the ShitSpotter repository.

Its purpose is to teach the discipline of turning hard, real engineering failures into reusable benchmark candidates and long-running project memory. Follow it when you make a nontrivial change, debug a difficult failure, touch dataset/model/evaluation logic, or discover a pattern that future agents are likely to miss.

This is not a TODO system. This is not user documentation. This is not a changelog. This is agent-readable engineering memory.

Recommended location in the repo:

```text
dev/AGENT_BENCHMARK_DISCIPLINE.md
```

Recommended companion tree:

```text
dev/
  README.md
  AGENT_BENCHMARK_DISCIPLINE.md
  benchmark-candidates/
    README.md
    cv-data-questions.md
    model-training-questions.md
    data-distribution-questions.md
    app-deployment-questions.md
    compositions.md
  journals/
    lessons_learned.md
```

---

## Core idea

A benchmark candidate is a distilled hard problem.

It is not merely a bug report. It is not merely a failing command. It is not a transcript of your work. It is a compact artifact that captures why a capable agent could plausibly fail, what invariant it must preserve, and how the answer can be checked.

For ShitSpotter, good benchmark candidates usually involve one or more of these invariants:

- kwcoco manifest integrity
- image / annotation / category id stability
- segmentation polygon, bbox, and mask consistency
- EXIF orientation, resize, crop, and letterbox geometry
- before / after / nearby-negative grouping semantics
- train / validation / test leakage prevention
- hard-negative and near-duplicate handling
- dataset versioning, IPFS CID, Hugging Face, Girder, and torrent mirror consistency
- model training reproducibility
- threshold, calibration, and F1 drift on new cohorts
- preprocessing parity between training, evaluation, export, and phone inference
- PyTorch / ONNX / TFLite / CoreML export parity
- small-data smoke checks that do not require the full dataset

A ShitSpotter benchmark candidate is not “can the agent write Python?” It is “can the agent preserve a data, model, scientific, or deployment invariant while doing a realistic maintenance task?”

---

## What belongs in `dev/`

Use `dev/` for long-running engineering memory that is useful to future agents.

Good entries:

- A difficult bug and the invariant that would have prevented it.
- A realistic benchmark question derived from an actual fix.
- A validation command that catches a subtle dataset/model mistake.
- A note about project structure that agents repeatedly infer incorrectly.
- A postmortem of an expensive debugging session.
- A small reproducible scenario that does not require private or huge data.
- A composition problem that requires coordinating multiple repo areas.

Bad entries:

- Ordinary TODOs.
- Generic project documentation.
- User-facing README material.
- Unverified guesses.
- Raw scratchpad dumps.
- Full transcripts.
- Huge logs with no distilled lesson.
- Claims that full dataset/model validation passed when only a smoke test ran.

If a note helps a future agent avoid wasting an hour, it probably belongs in `dev/`.

---

## When to create a benchmark candidate

Create or update a benchmark candidate when any of these happen:

1. You fix a bug that required more than local syntax reasoning.
2. You discover a repo-specific invariant not obvious from file names.
3. You catch an evaluation, dataset, annotation, or deployment mistake.
4. A change touches multiple domains: data, model, docs, packaging, release, app.
5. You needed to inspect several files before understanding the right fix.
6. You find that a plausible “simple” fix would be wrong.
7. A test failure was caused by hidden context, stale artifacts, split leakage, coordinate transforms, or external data/version assumptions.
8. You produce a new command, check, or minimal reproducer that future agents should reuse.

When in doubt, write a short candidate. A rough candidate can be refined later.

---

## Required workflow

Do not write benchmark candidates before understanding the failure.

Follow this order.

### 1. Fix or understand the real issue first

Before distilling the lesson, identify what actually went wrong.

Record:

- the original command, test, script, or workflow;
- the observed failure;
- the relevant files;
- the incorrect assumption;
- the final fix or best current understanding.

Do not invent a lesson that is not grounded in a real failure or a realistic repo maintenance task.

### 2. Preserve failure evidence

Save enough evidence that another agent could recognize the same class of failure.

Useful evidence includes:

- exact command;
- traceback or error snippet;
- relevant input file path;
- relevant config fragment;
- before/after manifest statistics;
- model/evaluation metric deltas;
- data split counts;
- artifact names, versions, or CIDs;
- small synthetic example.

Avoid pasting huge logs. Quote the smallest useful excerpt.

### 3. Reconstruct the pre-error context

A benchmark candidate should start from what an agent would have known before the fix.

Include:

- the task prompt or likely user request;
- the visible repo state;
- any misleading nearby files;
- the tempting but wrong solution;
- the hidden coupling that makes the problem hard.

The candidate should not depend on knowledge that only exists after reading the answer.

### 4. Distill the invariant

Name the invariant explicitly.

Examples:

```text
Invariant: kwcoco annotation ids must remain stable across manifest rewrites unless the task explicitly asks for reindexing.
```

```text
Invariant: Any image transform used during export must preserve the same normalization, resize, and channel-order semantics used during evaluation.
```

```text
Invariant: Before/after/nearby-negative images from the same collection group must not be split across train and validation.
```

If you cannot name the invariant, the candidate is probably not distilled enough.

### 5. Write the hard question

A good benchmark question asks for a realistic action, not a trivia answer.

Prefer:

```text
Given this failing kwcoco validation and this manifest-editing script, identify the invariant being violated, patch the script, and add a lightweight check.
```

Avoid:

```text
What is kwcoco?
```

The question should force the agent to reason across the relevant files or concepts.

### 6. Include the expected answer

Every benchmark candidate needs an answer key.

Include:

- expected diagnosis;
- expected file changes or pseudocode;
- expected validation command;
- why simpler fixes are wrong;
- what output or condition indicates success.

Do not hide the expected answer in prose scattered across paragraphs. Make it checkable.

### 7. Add validation

Every candidate should include at least one validation method.

Prefer validations that run without the full dataset:

- unit test;
- smoke test;
- synthetic kwcoco fixture;
- import check;
- schema or consistency check;
- grep/static check;
- tiny generated image/annotation example;
- command with explicit expected output.

If full data or trained weights are required, mark that clearly:

```text
Requires full dataset: yes
Requires trained weights: no
Lightweight substitute: tests/test_kwcoco_geometry.py::test_letterbox_polygon_roundtrip
```

Never claim full validation if only a smoke test ran.

---

## Candidate format

Use this template for entries in `dev/benchmark-candidates/*.md`.

```markdown
## <short descriptive title>

Status: draft | validated | promoted
Level: A | B | C
Tags: kwcoco, annotation-geometry, train-val-leakage, reproducibility
Requires full dataset: yes | no
Requires trained weights: yes | no

### Source context

Describe the real bug, change, or maintenance task this came from.

Include relevant files and commands.

### Pre-error setup

Describe what the agent sees before knowing the solution.

Include misleading assumptions or tempting wrong paths.

### Question

State the benchmark task as a clear prompt.

### Expected answer

Describe the correct diagnosis and expected change.

### Invariant

State the durable rule this benchmark is testing.

### Validation

List exact commands or checks.

### Wrong answers to reject

List plausible but incorrect fixes.

### Notes

Optional implementation details, links, or follow-up ideas.
```

---

## Levels

Use levels to communicate the type of reasoning required.

### Level A: single-invariant local problem

The agent must preserve one non-obvious invariant in a local change.

Examples:

- Fix a kwcoco annotation rewrite that changes ids.
- Correct a bbox/mask conversion bug.
- Add a smoke test for EXIF orientation handling.

### Level B: cross-file or cross-stage problem

The agent must connect two or more repo areas.

Examples:

- Training preprocessing and exported model preprocessing disagree.
- README dataset version is updated but CID/mirror metadata is stale.
- A validation split change silently leaks before/after pairs.

### Level C: composition problem

The agent must plan or execute a realistic maintenance workflow with multiple invariants.

Examples:

- Add a new annotated cohort while preserving split integrity, manifest consistency, dataset versioning, and evaluation comparability.
- Prepare a phone-deployable model release and verify export parity, threshold calibration, docs, and artifact metadata.
- Incorporate external Roboflow-style data while preserving license provenance, category mapping, duplicate handling, and evaluation isolation.

Composition problems are especially valuable. They test whether an agent can enumerate risks before acting.

---

## ShitSpotter-specific invariant checklist

Use this checklist when touching the relevant area.

### Dataset / kwcoco

Check:

- Are image ids stable?
- Are annotation ids stable unless intentionally reindexed?
- Are category ids and names stable?
- Are file names and relative paths valid?
- Are width/height correct after any transform?
- Are segmentation polygons valid and in image coordinates?
- Do bboxes enclose segmentations?
- Are empty or unannotated images represented intentionally?
- Are “before”, “after”, and “nearby negative” roles preserved?
- Are non-standard collection protocols explicitly marked?
- Can a small kwcoco validation script catch the change?

### Splits / evaluation

Check:

- Can related images leak across train/validation/test?
- Are near-duplicates grouped correctly?
- Are new cohorts assigned according to the current split policy?
- Are external datasets isolated from the primary test set when needed?
- Are metric deltas explained by data/model changes rather than split changes?
- Is the exact manifest path/version recorded in the experiment?

### Annotation geometry

Check:

- Are EXIF orientation and image dimensions handled consistently?
- Are resize, crop, pad, and letterbox transforms applied to polygons and boxes?
- Are masks, boxes, and polygons mutually consistent?
- Are coordinates clipped intentionally?
- Are degenerate polygons handled explicitly?
- Is there a round-trip or synthetic geometry test?

### Training / model artifacts

Check:

- Is the training config reproducible?
- Are seeds, splits, model weights, and dataset versions recorded?
- Is the threshold selected on the correct split?
- Is F1 or other detection metric computed with the intended IoU/confidence settings?
- Are bootstrap and release models clearly distinguished?
- Can a lightweight model-loading or config-parsing smoke test run?

### Distribution / release

Check:

- Are IPFS/IPNS/CID references consistent?
- Are Hugging Face, Girder, torrent, and README references consistent?
- Are stale links marked or removed?
- Is the latest dataset version unambiguous?
- Are model and dataset artifacts versioned separately?
- Is any external dataset license/provenance recorded?

### Phone / deployment

Check:

- Does exported inference use the same RGB/BGR order as evaluation?
- Does normalization match training?
- Does resize/letterbox behavior match evaluation?
- Are thresholds recalibrated after quantization or export?
- Is orientation handled for camera frames?
- Is the output coordinate system documented?
- Is there parity testing between PyTorch and exported inference on a tiny fixture?

---

## Journals

Use `dev/journals/lessons_learned.md` for expensive debugging sessions.

A journal entry is less formal than a benchmark candidate, but it should still be distilled.

Use this format:

```markdown
## YYYY-MM-DD: <short title>

### Trigger

What task or failure started the investigation?

### Symptoms

What was observed?

### Root cause

What was actually wrong?

### Fix

What changed?

### Durable lesson

What should future agents remember?

### Candidate follow-up

Should this become a benchmark candidate? Where?
```

If a journal entry produces a crisp invariant, also create a benchmark candidate.

---

## Agent behavior rules

When working in ShitSpotter, follow these rules.

1. Inspect before editing.
   Do not assume the repo has the same structure as another project.

2. Preserve scientific meaning.
   Dataset, split, annotation, and metric changes are not mere refactors.

3. Prefer small reproducible checks.
   When full data is unavailable, build a synthetic fixture that tests the invariant.

4. Do not overclaim validation.
   Clearly distinguish static checks, smoke tests, unit tests, partial data tests, and full dataset/model validation.

5. Separate public docs from agent memory.
   User-facing README/docs should explain the project. `dev/` should teach future agents how not to break it.

6. Distill after solving.
   Do not write vague “be careful” notes. Name the invariant and the validation.

7. Save the tempting wrong answer.
   Benchmarks are stronger when they document why a plausible shortcut fails.

8. Prefer invariant tags over surface-tool tags.
   `train-val-leakage` is better than `python-script`.
   `preprocessing-parity` is better than `onnx`.

9. Keep entries compact.
   A good candidate is usually one to three pages, not a raw transcript.

10. Update old candidates when reality changes.
    If the repo changes structure, update benchmark prompts so they remain realistic.

---

## Example seed candidate

```markdown
## Prevent split leakage across before/after/nearby-negative groups

Status: draft
Level: B
Tags: kwcoco, train-val-leakage, paired-before-after, hard-negative
Requires full dataset: no
Requires trained weights: no

### Source context

ShitSpotter images may be collected in related groups: a positive “before” image,
an “after” high-correlation negative, and a nearby lower-correlation negative.
A split script that treats images independently can leak scene context across
train and validation.

### Pre-error setup

An agent is asked to update the split after adding a new cohort. The manifest
contains enough metadata to infer or record grouping. A naive implementation
randomly assigns individual images.

### Question

Patch or design the split logic so related collection-group images cannot be
assigned to different splits. Add a lightweight validation that fails if any
group appears in more than one split.

### Expected answer

The split unit must be the collection group, not the individual image. The
implementation should group images by the repo’s chosen group key, assign the
group to a split once, then apply that assignment to all member images. The
validation should compute `group_id -> set(split)` and fail for any group with
more than one split.

### Invariant

Before/after/nearby-negative images from the same collection group must not be
split across train, validation, or test.

### Validation

Use a tiny synthetic kwcoco file with two groups and three images per group.
Run the split command or unit test and assert that each group has exactly one
split.

### Wrong answers to reject

- Randomly splitting individual images.
- Grouping only positive annotations while ignoring negatives.
- Checking only filename prefixes without verifying manifest metadata.
- Reporting aggregate split counts but not group leakage.
```

---

## Example composition prompt

Use this when asking an agent to perform a substantial ShitSpotter maintenance task.

```text
Before editing, enumerate the ShitSpotter invariants this task might affect.
For each invariant, name the file, test, script, or lightweight check that will
validate it. Then perform the smallest safe change. Afterward, update
`dev/benchmark-candidates/` or `dev/journals/lessons_learned.md` if the task
revealed a hard problem future agents should remember.

Do not copy patterns from another repo unless you first verify they fit the
ShitSpotter structure. Do not claim full dataset/model validation unless it
actually ran.
```

---

## Minimal end-of-task checklist

Before finishing a nontrivial task, answer these questions:

```text
1. Did I preserve the relevant dataset/model/scientific invariant?
2. Did I run a check that can catch the mistake I was worried about?
3. Did I clearly state what validation did and did not run?
4. Did I discover a hard problem future agents are likely to repeat?
5. If yes, did I add or update a benchmark candidate or journal entry?
```

If the answer to question 4 is yes and question 5 is no, the task is not complete.

