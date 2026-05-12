#!/bin/bash
# Print where each cell of the v4 Pareto sweep stands.
#
# Reads the on-disk state under $V4_ROOT and prints one row per
# candidate with what artifacts exist and (when available) their
# key numbers.
#
# Inputs (existence-checked per candidate):
#
#   $V4_ROOT/runs/<candidate_id>/best_stg2.pth       trained
#   $V4_ROOT/runs/<candidate_id>/checkpoint*.pth     last-epoch ckpt
#   $V4_ROOT/runs/<candidate_id>/export/*.onnx       ONNX exported
#   $V4_ROOT/runs/<candidate_id>/export/*.bench.json desktop bench
#   $V4_ROOT/eval/<candidate_id>/eval/detect_metrics.json   test AP
#   $V4_ROOT/sweeps/<UTC>/index.tsv                  per-sweep manifest
#
# Usage:
#   source experiments/mobile_app_training_v4/setup_env.sh
#   bash   experiments/mobile_app_training_v4/08_status.sh
#
#   # Or constrain to a specific sweep TSV:
#   bash   experiments/mobile_app_training_v4/08_status.sh \
#          "$V4_ROOT/sweeps/20260512T000000Z/index.tsv"

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
# shellcheck source=experiments/mobile_app_training_v4/setup_env.sh
source "$_v4_script_dpath/setup_env.sh" >/dev/null
V4_SCRIPT_DPATH="$_v4_script_dpath"
unset _v4_source _v4_script_dpath

SWEEP_INDEX="${1:-}"
# Resolve a working python interpreter: PYTHON_BIN, then python, then python3.
PY="${PYTHON_BIN:-}"
if [ -z "$PY" ] || ! command -v "$PY" >/dev/null 2>&1; then
    if command -v python  >/dev/null 2>&1; then PY=python
    elif command -v python3 >/dev/null 2>&1; then PY=python3
    else echo "no python interpreter found on PATH" >&2; exit 1
    fi
fi

if [ ! -d "$V4_ROOT" ]; then
    echo "V4_ROOT does not exist: $V4_ROOT" >&2
    exit 1
fi

# If no sweep TSV passed, auto-pick the most recent one (if any) and
# also scan all of $V4_ROOT/runs/ so we report cells that exist on disk
# but never made it into a sweep index.
if [ -z "$SWEEP_INDEX" ]; then
    if compgen -G "$V4_ROOT/sweeps/*/index.tsv" > /dev/null; then
        SWEEP_INDEX="$(ls -1t "$V4_ROOT"/sweeps/*/index.tsv | head -1)"
    fi
fi

echo "=== mobile_app_training_v4 / sweep status ==="
printf '  %-32s %s\n' "V4_ROOT"      "$V4_ROOT"
printf '  %-32s %s\n' "SWEEP_INDEX"  "${SWEEP_INDEX:-<none found; scanning runs/ only>}"
echo

"$PY" - "$V4_ROOT" "${SWEEP_INDEX:-}" <<'PY'
import json, os, sys, glob, csv

v4_root = sys.argv[1]
sweep_idx = sys.argv[2] or None

def human_bytes(n):
    if n is None: return '-'
    for unit, scale in (('G', 1<<30), ('M', 1<<20), ('K', 1<<10)):
        if n >= scale:
            return f'{n/scale:.1f}{unit}'
    return f'{n}B'

def find_ap(node):
    if isinstance(node, dict):
        if 'nocls_measures' in node and isinstance(node['nocls_measures'], dict):
            v = node['nocls_measures'].get('ap')
            if v is not None: return float(v)
        for v in node.values():
            r = find_ap(v);
            if r is not None: return r
    elif isinstance(node, list):
        for v in node:
            r = find_ap(v)
            if r is not None: return r
    return None

# Build candidate list: union of (a) sweep index rows and (b) runs/ dirs.
candidates = []  # list of dicts with at least candidate_id, variant, h, w, policy
seen = set()

if sweep_idx and os.path.isfile(sweep_idx):
    with open(sweep_idx, newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            cid = row.get('candidate_id') or ''
            if not cid or cid in seen: continue
            seen.add(cid)
            candidates.append({
                'candidate_id': cid,
                'variant':      row.get('variant', ''),
                'h':            row.get('export_h', ''),
                'w':            row.get('export_w', ''),
                'policy':       row.get('train_policy', ''),
                'sweep_status': row.get('status', ''),
            })

for d in sorted(glob.glob(os.path.join(v4_root, 'runs', '*'))):
    cid = os.path.basename(d)
    if cid in seen: continue
    seen.add(cid)
    candidates.append({
        'candidate_id': cid, 'variant': '', 'h': '', 'w': '',
        'policy': '', 'sweep_status': '',
    })

if not candidates:
    print('No candidates found — nothing under $V4_ROOT/runs/ and no sweep index.')
    sys.exit(0)

# One-line-per-cell report
fmt = '{cid:<48}  {train:<5}  {onnx:<10}  {eval_ap:<12}  {bench:<14}  {sweep}'
print(fmt.format(cid='candidate_id', train='train', onnx='onnx',
                 eval_ap='eval_AP', bench='bench_mean_ms', sweep='sweep_status'))
print('-' * 110)

for c in candidates:
    cid = c['candidate_id']
    workdir = os.path.join(v4_root, 'runs', cid)
    export_d = os.path.join(workdir, 'export')

    # Training: any of best_stg2.pth, best_stg1.pth, last.pth, checkpointN.pth.
    train = '-'
    for name in ('best_stg2.pth', 'best_stg1.pth', 'last.pth'):
        if os.path.isfile(os.path.join(workdir, name)):
            train = name.replace('.pth', ''); break
    if train == '-':
        for cp in sorted(glob.glob(os.path.join(workdir, 'checkpoint*.pth'))):
            train = 'last_ckpt'; break

    # ONNX: take the largest .onnx in export/
    onnx = '-'
    onnx_paths = sorted(glob.glob(os.path.join(export_d, '*.onnx')))
    if onnx_paths:
        best = max(onnx_paths, key=lambda p: os.path.getsize(p))
        onnx = human_bytes(os.path.getsize(best))
        # Flag obviously stub / tiny exports.
        if os.path.getsize(best) < 256 * 1024:
            onnx += '!'

    # Eval AP
    ev = os.path.join(v4_root, 'eval', cid, 'eval', 'detect_metrics.json')
    ap = None
    if os.path.isfile(ev):
        try:
            ap = find_ap(json.loads(open(ev).read()))
        except Exception:
            ap = None
    eval_ap = f'{ap:.4f}' if ap is not None else '-'

    # Bench
    bench = '-'
    for bj in glob.glob(os.path.join(export_d, '*.bench.json')):
        try:
            d = json.loads(open(bj).read())
            v = d.get('mean_ms')
            if v is not None: bench = f'{float(v):.1f}'
            break
        except Exception:
            pass

    print(fmt.format(cid=cid, train=train, onnx=onnx, eval_ap=eval_ap,
                     bench=bench, sweep=c['sweep_status'] or '-'))

print()
print('legend: train ∈ {best_stg2,best_stg1,last,last_ckpt,-}  '
      'onnx size suffix ! = under 256K (probable stub)')
print('eval_AP is the simplified-GT AP @ IoU=0.5 (v9 reference = 0.766)')
PY

echo
echo "Aggregate eligibility (writes \$V4_ROOT/manifest.tsv):"
echo "  \"$PY\" \"$V4_SCRIPT_DPATH/eligibility_manifest.py\" \\"
echo "      --auto --max_desktop_ms 80 \\"
echo "      --out      \"$V4_ROOT/manifest.tsv\" \\"
echo "      --out_json \"$V4_ROOT/manifest.json\""
