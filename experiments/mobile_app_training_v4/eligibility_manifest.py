#!/usr/bin/env python3
"""
Aggregate the per-cell outputs of a sweep into a single eligibility
manifest (TSV + JSON), and select the highest-quality candidate that
satisfies the deployment gates.

This is the explicit selection rule the v4 design calls for: a model is
not chosen because it is smallest or fastest. It is chosen because it
is the highest-quality candidate that meets the Pixel 5 gate.

Three eligibility classes (no candidate is "deploy-eligible" until
Pixel 5 data is supplied):

  HOST_PROMISING    passes host-side gates: training, ONNX export, kwcoco
                    eval, AND desktop CPU latency <= --max_desktop_ms.
                    Worth flashing onto a phone for real measurement.

  PHONE_ELIGIBLE    HOST_PROMISING AND --pixel5_index supplied AND
                    pixel5_fps >= --min_pixel5_fps. Only this class is
                    safe to call "eligible to deploy."

  PHONE_INELIGIBLE  --pixel5_index supplied and pixel5_fps below the
                    gate, OR a host-side gate failed.

When --pixel5_index is missing, no candidate is PHONE_ELIGIBLE — the
script prints a "host-promising winner" and explicitly says no deploy-
eligible winner can be selected without on-device data.

Per-candidate fields recorded:

  candidate_id, variant,
  export_input_h, export_input_w,
  train_resolution_policy, train_resolution_min, train_resolution_max,
  train_resolution_choices,
  tile_training_policy,
  checkpoint_path, onnx_path, modelspec_path,
  test_ap_simplified,
  desktop_latency_ms_p50, desktop_latency_ms_mean, desktop_latency_ms_p99,
  desktop_eligible,
  pixel5_latency_ms, pixel5_fps, pixel5_eligible,
  phone_model_id, status, reasons

Usage::

    python eligibility_manifest.py \\
        --sweep_index $V4_ROOT/sweeps/<TS>/index.tsv \\
        --max_desktop_ms 80 \\
        --out         $V4_ROOT/sweeps/<TS>/manifest.tsv

Or aggregate everything under $V4_ROOT/runs/ without an index TSV::

    python eligibility_manifest.py --auto --out $V4_ROOT/manifest.tsv
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import scriptconfig as scfg


MANIFEST_FIELDS = [
    'candidate_id',
    'variant',
    'export_input_h',
    'export_input_w',
    'train_resolution_policy',
    'requested_train_resolution_min',
    'requested_train_resolution_max',
    'train_resolution_min',
    'train_resolution_max',
    'train_resolution_choices',
    'tile_training_policy',
    'checkpoint_path',
    'onnx_path',
    'modelspec_path',
    'test_ap_simplified',
    'desktop_latency_ms_p50',
    'desktop_latency_ms_mean',
    'desktop_latency_ms_p99',
    'desktop_eligible',
    'pixel5_latency_ms',
    'pixel5_fps',
    'pixel5_eligible',
    'eligibility_class',
    'phone_model_id',
    'status',
    'reasons',
]


@dataclass
class Row:
    candidate_id: str = ''
    variant: str = ''
    export_input_h: Optional[int] = None
    export_input_w: Optional[int] = None
    train_resolution_policy: str = ''
    requested_train_resolution_min: Optional[int] = None
    requested_train_resolution_max: Optional[int] = None
    train_resolution_min: Optional[int] = None
    train_resolution_max: Optional[int] = None
    train_resolution_choices: str = ''
    tile_training_policy: str = ''
    checkpoint_path: str = ''
    onnx_path: str = ''
    modelspec_path: str = ''
    test_ap_simplified: Optional[float] = None
    desktop_latency_ms_p50: Optional[float] = None
    desktop_latency_ms_mean: Optional[float] = None
    desktop_latency_ms_p99: Optional[float] = None
    desktop_eligible: str = ''            # 'yes' | 'no' | ''
    pixel5_latency_ms: Optional[float] = None
    pixel5_fps: Optional[float] = None
    pixel5_eligible: str = 'TODO'
    eligibility_class: str = ''           # HOST_PROMISING | PHONE_ELIGIBLE | PHONE_INELIGIBLE | NOT_READY
    phone_model_id: str = ''
    status: str = ''
    reasons: list = field(default_factory=list)


class ManifestCLI(scfg.DataConfig):
    sweep_index = scfg.Value(None, help='TSV index produced by 02_sweep.sh; takes precedence over --auto')
    auto = scfg.Value(False, isflag=True, help='Discover candidates by walking $V4_ROOT/runs/')
    v4_root = scfg.Value(None, help='V4_ROOT to scan in --auto mode (defaults to $V4_ROOT or ~/data/shitspotter_v4)')
    out = scfg.Value(None, help='output TSV path')
    out_json = scfg.Value(None, help='optional output JSON path (richer; lists train scales etc.)')
    max_desktop_ms = scfg.Value(80.0, help='desktop CPU mean ms gate (proxy for the on-device gate)')
    min_pixel5_fps = scfg.Value(10.0, help='Pixel 5 FPS gate (only enforced when --pixel5_index is given)')
    pixel5_index = scfg.Value(None, help='optional TSV with columns candidate_id\\tlatency_ms\\tfps')
    print_winner = scfg.Value(True, isflag=True, help='print the eligible winner to stdout')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


# ---------- discovery ------------------------------------------------------

def _iter_candidates_from_index(index_fpath):
    with open(index_fpath, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('status', '').strip() in ('fail', 'skip'):
                continue
            yield row.get('candidate_id', '').strip(), Path(row.get('workdir', '').strip())


def _iter_candidates_from_auto(v4_root):
    runs_root = Path(v4_root) / 'runs'
    if not runs_root.exists():
        return
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        policy_fpath = sub / 'policy.json'
        if not policy_fpath.exists():
            continue
        # Prefer the policy.json's own candidate_id (this is the same
        # identity 03_export_onnx.sh embedded in the modelspec), so the
        # eval/<candidate_id> directory lookup matches.
        try:
            cid = json.loads(policy_fpath.read_text()).get('candidate_id') or sub.name
        except Exception:
            cid = sub.name
        yield cid, sub


# ---------- per-candidate readers ------------------------------------------

def _load_policy(workdir: Path) -> dict:
    fpath = workdir / 'policy.json'
    if not fpath.exists():
        return {}
    try:
        return json.loads(fpath.read_text())
    except Exception as ex:
        return {'_policy_load_error': str(ex)}


def _find_checkpoint(workdir: Path) -> str:
    for cand in ('best_stg2.pth', 'best_stg1.pth', 'last.pth'):
        if (workdir / cand).exists():
            return str(workdir / cand)
    epochs = sorted(workdir.glob('checkpoint*.pth'))
    return str(epochs[-1]) if epochs else ''


def _find_onnx_and_modelspec(workdir: Path) -> tuple:
    export_dpath = workdir / 'export'
    if not export_dpath.exists():
        return '', ''
    onnx_files = sorted(export_dpath.glob('*.onnx'))
    if not onnx_files:
        return '', ''
    onnx = onnx_files[0]
    modelspec = onnx.with_suffix('.modelspec.json')
    return str(onnx), (str(modelspec) if modelspec.exists() else '')


def _find_eval_ap(v4_root: Path, candidate_id: str) -> Optional[float]:
    metrics_fpath = v4_root / 'eval' / candidate_id / 'eval' / 'detect_metrics.json'
    if not metrics_fpath.exists():
        return None
    try:
        data = json.loads(metrics_fpath.read_text())
    except Exception:
        return None

    def find_ap(node):
        if isinstance(node, dict):
            if 'nocls_measures' in node and isinstance(node['nocls_measures'], dict):
                v = node['nocls_measures'].get('ap')
                if v is not None:
                    return float(v)
            for v in node.values():
                r = find_ap(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for v in node:
                r = find_ap(v)
                if r is not None:
                    return r
        return None
    return find_ap(data)


def _find_bench_metrics(workdir: Path) -> dict:
    export_dpath = workdir / 'export'
    if not export_dpath.exists():
        return {}
    candidates = sorted(export_dpath.glob('*.bench.json'))
    if not candidates:
        return {}
    try:
        bench = json.loads(candidates[0].read_text())
    except Exception:
        return {}
    timings = bench.get('timings_ms', [])
    if not timings:
        return {'mean_ms': bench.get('mean_ms')}
    timings_sorted = sorted(timings)
    def pct(p):
        idx = max(0, min(len(timings_sorted) - 1, int(round(p / 100.0 * (len(timings_sorted) - 1)))))
        return timings_sorted[idx]
    return {
        'mean_ms': bench.get('mean_ms', sum(timings) / len(timings)),
        'p50_ms': pct(50),
        'p99_ms': pct(99),
    }


def _phone_model_id(policy: dict) -> str:
    """Canonical phone-app model ID. Must stay in lock-step with the
    same string built by 03_export_onnx.sh and written to the
    .modelspec.json sidecar (`modelId` field)."""
    v = policy.get('variant', '')
    h = policy.get('export_input_h', '')
    w = policy.get('export_input_w', '')
    pol = policy.get('train_resolution_policy', '')
    return f'shitspotter-{v}-h{h}w{w}-{pol}'


# ---------- pixel5 sidecar -------------------------------------------------

def _load_pixel5_index(fpath):
    if fpath is None:
        return {}
    out = {}
    with open(str(fpath), 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            cid = row.get('candidate_id', '').strip()
            if not cid:
                continue
            out[cid] = {
                'latency_ms': _safe_float(row.get('latency_ms')),
                'fps': _safe_float(row.get('fps')),
            }
    return out


def _safe_float(v):
    try:
        return float(v) if v not in (None, '', 'None') else None
    except Exception:
        return None


# ---------- main -----------------------------------------------------------

def run(config):
    import os

    v4_root = Path(config.v4_root or os.environ.get('V4_ROOT')
                   or (Path.home() / 'data' / 'shitspotter_v4'))
    pixel5 = _load_pixel5_index(config.pixel5_index)

    if config.sweep_index:
        cand_iter = list(_iter_candidates_from_index(config.sweep_index))
    elif config.auto:
        cand_iter = list(_iter_candidates_from_auto(v4_root))
    else:
        raise SystemExit('Either --sweep_index or --auto is required')

    rows = []
    for candidate_id, workdir in cand_iter:
        if not workdir.exists():
            print(f'  skip: {candidate_id} workdir missing ({workdir})', file=sys.stderr)
            continue
        policy = _load_policy(workdir)
        if not candidate_id:
            candidate_id = policy.get('candidate_id', workdir.name)

        ckpt = _find_checkpoint(workdir)
        onnx, modelspec = _find_onnx_and_modelspec(workdir)
        ap = _find_eval_ap(v4_root, candidate_id)
        bench = _find_bench_metrics(workdir)

        scales = policy.get('effective_train_scales', [])
        row = Row(
            candidate_id=candidate_id,
            variant=policy.get('variant', ''),
            export_input_h=policy.get('export_input_h'),
            export_input_w=policy.get('export_input_w'),
            train_resolution_policy=policy.get('train_resolution_policy', ''),
            requested_train_resolution_min=policy.get('requested_train_resolution_min'),
            requested_train_resolution_max=policy.get('requested_train_resolution_max'),
            train_resolution_min=policy.get('effective_train_scale_min'),
            train_resolution_max=policy.get('effective_train_scale_max'),
            train_resolution_choices=','.join(str(s) for s in scales),
            tile_training_policy=policy.get('tile_training_policy', ''),
            checkpoint_path=ckpt,
            onnx_path=onnx,
            modelspec_path=modelspec,
            test_ap_simplified=ap,
            desktop_latency_ms_p50=bench.get('p50_ms'),
            desktop_latency_ms_mean=bench.get('mean_ms'),
            desktop_latency_ms_p99=bench.get('p99_ms'),
            phone_model_id=_phone_model_id(policy),
        )

        # Status gate (was the host-side pipeline able to produce something
        # we can even consider?)
        reasons = []
        if not ckpt:
            row.status = 'no_checkpoint'; reasons.append('no .pth in workdir')
        elif not onnx:
            row.status = 'no_onnx'; reasons.append('no exported .onnx')
        elif ap is None:
            row.status = 'no_eval'; reasons.append('no test detect_metrics.json')
        else:
            row.status = 'ok'

        # Desktop proxy gate (latency)
        mean_ms = row.desktop_latency_ms_mean
        if mean_ms is None:
            row.desktop_eligible = ''
        elif mean_ms <= float(config.max_desktop_ms):
            row.desktop_eligible = 'yes'
        else:
            row.desktop_eligible = 'no'
            reasons.append(f'desktop mean {mean_ms:.1f}ms > {config.max_desktop_ms}ms')

        # On-device gate (only meaningful when --pixel5_index is supplied)
        p5 = pixel5.get(candidate_id)
        if p5 is not None:
            row.pixel5_latency_ms = p5.get('latency_ms')
            row.pixel5_fps = p5.get('fps')
            if row.pixel5_fps is not None and row.pixel5_fps >= float(config.min_pixel5_fps):
                row.pixel5_eligible = 'yes'
            else:
                row.pixel5_eligible = 'no'
                reasons.append(f'pixel5 {row.pixel5_fps} fps < {config.min_pixel5_fps}')
        else:
            row.pixel5_eligible = 'TODO'

        # Three-class eligibility — never call anything PHONE_ELIGIBLE
        # without on-device data. NOT_READY = host pipeline broken.
        # HOST_PROMISING = passes host gates, awaiting phone validation.
        if row.status != 'ok':
            row.eligibility_class = 'NOT_READY'
        elif row.desktop_eligible == 'no':
            row.eligibility_class = 'PHONE_INELIGIBLE'  # desktop proxy already says no
        elif row.pixel5_eligible == 'no':
            row.eligibility_class = 'PHONE_INELIGIBLE'
        elif row.pixel5_eligible == 'yes':
            row.eligibility_class = 'PHONE_ELIGIBLE'
        else:
            # status ok, desktop proxy ok-or-blank, pixel5 untested
            row.eligibility_class = 'HOST_PROMISING'

        row.reasons = reasons
        rows.append(row)

    # ---- write outputs ---------------------------------------------------
    out_fpath = Path(str(config.out)) if config.out else None
    if out_fpath:
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        with out_fpath.open('w', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerow(MANIFEST_FIELDS)
            for row in rows:
                d = asdict(row)
                d['reasons'] = '; '.join(row.reasons)
                w.writerow([d.get(k, '') if d.get(k) is not None else '' for k in MANIFEST_FIELDS])
        print(f'wrote {out_fpath}')

    if config.out_json:
        Path(str(config.out_json)).parent.mkdir(parents=True, exist_ok=True)
        Path(str(config.out_json)).write_text(json.dumps([asdict(r) for r in rows], indent=2))

    # ---- print summary table -------------------------------------------
    print()
    print('candidate_id'.ljust(50),
          'AP'.rjust(7),
          'desk_ms'.rjust(8),
          'desk_ok'.rjust(8),
          'p5_fps'.rjust(7),
          'p5_ok'.rjust(6),
          'class'.rjust(17),
          'status')
    print('-' * 130)
    for row in sorted(rows, key=lambda r: (r.test_ap_simplified or -1), reverse=True):
        ap = '-' if row.test_ap_simplified is None else f'{row.test_ap_simplified:.3f}'
        ms = '-' if row.desktop_latency_ms_mean is None else f'{row.desktop_latency_ms_mean:.1f}'
        fps = '-' if row.pixel5_fps is None else f'{row.pixel5_fps:.1f}'
        print(row.candidate_id.ljust(50),
              ap.rjust(7),
              ms.rjust(8),
              row.desktop_eligible.rjust(8),
              fps.rjust(7),
              row.pixel5_eligible.rjust(6),
              row.eligibility_class.rjust(17),
              row.status)

    if not config.print_winner:
        return

    # ---- selection ------------------------------------------------------
    # Two distinct concepts the reviewer flagged:
    #
    #   host-promising winner   highest AP among HOST_PROMISING — worth
    #                           sideloading. Not deploy-eligible.
    #
    #   deploy-eligible winner  highest AP among PHONE_ELIGIBLE.
    #                           Only printed when --pixel5_index supplied
    #                           and at least one cell passes the FPS gate.
    has_pixel5 = bool(pixel5)
    promising = [r for r in rows
                 if r.eligibility_class == 'HOST_PROMISING'
                 and r.test_ap_simplified is not None]
    eligible = [r for r in rows
                if r.eligibility_class == 'PHONE_ELIGIBLE'
                and r.test_ap_simplified is not None]

    def _print_winner(label, row):
        print()
        print(f'=== {label} ===')
        print(f'  candidate_id          {row.candidate_id}')
        print(f'  variant               {row.variant}')
        print(f'  export size           {row.export_input_h} x {row.export_input_w}')
        print(f'  train policy          {row.train_resolution_policy}'
              f'  scales={row.train_resolution_min}..{row.train_resolution_max}')
        print(f'  test AP (simplified)  {row.test_ap_simplified:.3f}'
              f'   (v9 baseline = 0.766)')
        if row.desktop_latency_ms_mean is not None:
            print(f'  desktop CPU mean      {row.desktop_latency_ms_mean:.1f} ms')
        if row.pixel5_fps is not None:
            print(f'  Pixel 5 fps           {row.pixel5_fps:.1f}')
        print(f'  phone_model_id        {row.phone_model_id}')
        print(f'  onnx                  {row.onnx_path}')
        print(f'  modelspec             {row.modelspec_path}')

    if promising:
        _print_winner('host-promising winner', max(promising, key=lambda r: r.test_ap_simplified))
    else:
        print('\nno HOST_PROMISING candidate yet — fix the failing gates above')

    if has_pixel5:
        if eligible:
            _print_winner('deploy-eligible winner', max(eligible, key=lambda r: r.test_ap_simplified))
        else:
            print('\nno PHONE_ELIGIBLE candidate — every cell failed the Pixel 5 gate')
    else:
        print('\nno deploy-eligible winner can be selected without --pixel5_index;')
        print('run on-device benchmarks and supply a candidate_id\\tlatency_ms\\tfps TSV')


__cli__ = ManifestCLI


if __name__ == '__main__':
    __cli__.main()
