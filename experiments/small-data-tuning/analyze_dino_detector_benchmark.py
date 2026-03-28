#!/usr/bin/env python3
"""
Aggregate benchmark summaries and draw train-size curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import re


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_root', required=True)
    parser.add_argument('--out_dpath', required=True)
    return parser


def _read_summary_rows(summary_fpath: pathlib.Path) -> list[dict]:
    with summary_fpath.open('r', newline='') as file:
        return list(csv.DictReader(file, delimiter='\t'))


def _coerce_float(text):
    if text in {None, '', 'NA', 'NaN', 'nan'}:
        return None
    return float(text)


def _is_finite_number(value) -> bool:
    return value is not None and math.isfinite(value)


def _row_preference_key(row: dict) -> tuple:
    # Prefer rows with more complete metrics, then prefer tagged-layout runs
    # over legacy baseline-layout runs when the scientific identity is the same.
    run_path = pathlib.Path(row.get('run_dpath', ''))
    rel_depth = len(run_path.parts)
    return (
        1 if _is_finite_number(row.get('poop_test_ap')) else 0,
        1 if _is_finite_number(row.get('poop_vali_ap')) else 0,
        rel_depth,
    )


def _load_json_if_exists(fpath: pathlib.Path):
    if fpath.exists():
        return json.loads(fpath.read_text())
    return None


def _parse_python_batch_size(config_fpath: pathlib.Path):
    if not config_fpath.exists():
        return None
    text = config_fpath.read_text()
    match = re.search(r'^\s*batch_size\s*=\s*([0-9]+)\s*$', text, flags=re.MULTILINE)
    if match:
        return int(match.group(1))
    return None


def _augment_with_run_metadata(row: dict) -> dict:
    row = dict(row)
    run_dpath = pathlib.Path(row.get('run_dpath', ''))
    run_manifest_fpath = run_dpath / 'run_manifest.json'
    row['run_manifest_fpath'] = str(run_manifest_fpath) if run_manifest_fpath.exists() else ''
    row['train_batch_size'] = ''
    row['vali_batch_size'] = ''
    row['num_gpus_requested'] = ''
    manifest = _load_json_if_exists(run_manifest_fpath)
    if manifest:
        if 'train_batch_size' in manifest:
            row['train_batch_size'] = manifest.get('train_batch_size', '')
        if 'vali_batch_size' in manifest:
            row['vali_batch_size'] = manifest.get('vali_batch_size', '')
        if 'num_gpus_requested' in manifest:
            row['num_gpus_requested'] = manifest.get('num_gpus_requested', '')
        elif 'gpu_num' in manifest:
            row['num_gpus_requested'] = manifest.get('gpu_num', '')

    if row.get('model_family') == 'opengroundingdino' and not row['train_batch_size']:
        config_json = run_dpath / 'train_output' / 'config_args_all.json'
        config_data = _load_json_if_exists(config_json)
        if config_data and 'batch_size' in config_data:
            row['train_batch_size'] = config_data.get('batch_size', '')
        else:
            cfg_py = run_dpath / 'prepared' / 'shitspotter_cfg_odvg.py'
            parsed_batch_size = _parse_python_batch_size(cfg_py)
            if parsed_batch_size is not None:
                row['train_batch_size'] = parsed_batch_size

    return row


def _safe_link_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text)


def _refresh_symlink(link_fpath: pathlib.Path, target: pathlib.Path) -> None:
    link_fpath.parent.mkdir(parents=True, exist_ok=True)
    if link_fpath.is_symlink() or link_fpath.exists():
        if link_fpath.is_symlink() and os.path.realpath(link_fpath) == str(target.resolve()):
            return
        if link_fpath.is_dir() and not link_fpath.is_symlink():
            return
        link_fpath.unlink()
    link_fpath.symlink_to(target)


def _normalize_row(row: dict, *, model_family: str, subset_name: str) -> dict:
    row = dict(row)
    row['model_family'] = model_family
    row['subset_name'] = subset_name
    row.setdefault('config_tag', 'baseline')
    row['config_label'] = f"{model_family}:{row['config_tag']}"
    row.setdefault('status', 'selected')
    row['selected_candidate_id'] = row.get('candidate_id', '')
    row['train_size'] = int(row['train_size'])
    row['poop_vali_ap'] = _coerce_float(row.get('poop_vali_ap'))
    row['poop_test_ap'] = _coerce_float(row.get('poop_test_ap'))
    row['nocls_vali_ap'] = _coerce_float(row.get('nocls_vali_ap'))
    row['nocls_test_ap'] = _coerce_float(row.get('nocls_test_ap'))
    return row


def _generate_plots(selected_rows: list[dict], plots_dpath: pathlib.Path,
                     legacy_plot_fpath: pathlib.Path) -> None:
    """Generate multi-variant plots using kwplot utilities.

    Produces:
        PLT01_train_size_curve_legend.png     - curve plot with legend
        PLT02_train_size_curve_nolegend.png   - curve plot without legend
        PLT03_train_size_curve_onlylegend.png - standalone legend
        PLT04_results_table.png               - summary table as image
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import kwplot
    import pandas as pd

    finalize = kwplot.FigureFinalizer(
        dpi=200,
        cropwhite=True,
    )

    # --- Build config_label palette for consistent colors ---
    config_labels = sorted({row['config_label'] for row in selected_rows})
    from kwplot.managers import Palette
    palette = Palette.coerce(config_labels)

    # --- Draw curves: one color per config_label, solid=vali, dashed=test ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for config_label in config_labels:
        color = palette[config_label]
        family_rows = sorted(
            [
                row for row in selected_rows
                if row['config_label'] == config_label
                and _is_finite_number(row['poop_vali_ap'])
            ],
            key=lambda row: row['train_size'],
        )
        if not family_rows:
            continue
        ax.plot(
            [row['train_size'] for row in family_rows],
            [row['poop_vali_ap'] for row in family_rows],
            marker='o', color=color,
            label=f'{config_label} vali',
        )
        test_rows = [row for row in family_rows if _is_finite_number(row['poop_test_ap'])]
        if test_rows:
            ax.plot(
                [row['train_size'] for row in test_rows],
                [row['poop_test_ap'] for row in test_rows],
                marker='s', linestyle='--', color=color,
                label=f'{config_label} test',
            )
    ax.set_xlabel('Training size')
    ax.set_ylabel('Box AP (poop)')
    ax.set_title('DINO detector benchmark')
    ax.grid(True, alpha=0.3)

    # PLT01: with legend
    ax.legend(fontsize='small', loc='best')
    fpath_legend = plots_dpath / 'PLT01_train_size_curve_legend.png'
    finalize.finalize(fig, fpath_legend)

    # PLT03: standalone legend (extract before removing)
    try:
        legend_ax = kwplot.extract_legend(ax)
        fpath_onlylegend = plots_dpath / 'PLT03_train_size_curve_onlylegend.png'
        finalize.finalize(legend_ax.figure, fpath_onlylegend)
        plt.close(legend_ax.figure)
    except Exception:
        pass  # non-critical

    # PLT02: without legend
    legend_obj = ax.get_legend()
    if legend_obj is not None:
        legend_obj.remove()
    fpath_nolegend = plots_dpath / 'PLT02_train_size_curve_nolegend.png'
    finalize.finalize(fig, fpath_nolegend)

    # Copy to legacy path
    import shutil
    shutil.copy2(fpath_legend, legacy_plot_fpath)

    plt.close(fig)

    # PLT04: results table as image
    table_rows = []
    for row in selected_rows:
        if row.get('status') == 'no_finite_selection':
            continue
        table_rows.append({
            'model': row.get('model_family', ''),
            'config': row.get('config_tag', ''),
            'train_size': row.get('train_size', ''),
            'batch': row.get('train_batch_size', ''),
            'best_ckpt': row.get('selected_candidate_id', ''),
            'vali_ap': row.get('poop_vali_ap'),
            'test_ap': row.get('poop_test_ap'),
        })
    if table_rows:
        df = pd.DataFrame(table_rows)
        df = df.sort_values(['model', 'config', 'train_size'])
        # Format AP columns
        for col in ['vali_ap', 'test_ap']:
            df[col] = df[col].apply(
                lambda v: f'{v:.4f}' if _is_finite_number(v) else ''
            )
        fpath_table = plots_dpath / 'PLT04_results_table.png'
        try:
            kwplot.dataframe_table(df, fpath_table, title='DINO Detector Benchmark')
            print(f'  TABLE_FPATH            {fpath_table}')
        except Exception as ex:
            print(f'  TABLE_FPATH            (skipped: {ex})')


def main(argv=1) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv if isinstance(argv, list) else None)
    benchmark_root = pathlib.Path(args.benchmark_root)
    out_dpath = pathlib.Path(args.out_dpath)
    out_dpath.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    summary_fpaths = sorted(benchmark_root.glob('runs/*/*/summary.tsv'))
    summary_fpaths += sorted(benchmark_root.glob('runs/*/*/*/summary.tsv'))
    seen = set()
    deduped_summary_fpaths = []
    for summary_fpath in summary_fpaths:
        if summary_fpath in seen:
            continue
        deduped_summary_fpaths.append(summary_fpath)
        seen.add(summary_fpath)

    for summary_fpath in deduped_summary_fpaths:
        rel_parts = summary_fpath.relative_to(benchmark_root).parts
        if len(rel_parts) == 4:
            _, model_family, subset_name, _ = rel_parts
            config_tag = 'baseline'
        elif len(rel_parts) == 5:
            _, model_family, config_tag, subset_name, _ = rel_parts
        else:
            continue
        raw_rows = _read_summary_rows(summary_fpath)
        selected_row = None
        for row in raw_rows:
            if str(row.get('selected', '')).strip() == '1':
                selected_row = _normalize_row(row, model_family=model_family, subset_name=subset_name)
                selected_row['config_tag'] = config_tag
                selected_row['config_label'] = f'{model_family}:{config_tag}'
                break
        if selected_row is None or not _is_finite_number(selected_row.get('poop_vali_ap')):
            finite_candidates = [
                _normalize_row(row, model_family=model_family, subset_name=subset_name)
                for row in raw_rows
                if _is_finite_number(_coerce_float(row.get('poop_vali_ap')))
            ]
            if finite_candidates:
                selected_row = max(finite_candidates, key=lambda row: row['poop_vali_ap'])
                selected_row['config_tag'] = config_tag
                selected_row['config_label'] = f'{model_family}:{config_tag}'
                selected_row['selected_candidate_id'] = selected_row.get('candidate_id', '')
                selected_row['status'] = 'selection_recovered'
                selected_row['selection_recovered'] = '1'
                selected_row['poop_test_ap'] = None
                selected_row['nocls_test_ap'] = None
        if selected_row is None:
            if raw_rows:
                placeholder = _normalize_row(raw_rows[0], model_family=model_family, subset_name=subset_name)
                placeholder['config_tag'] = config_tag
                placeholder['config_label'] = f'{model_family}:{config_tag}'
                placeholder['selected_candidate_id'] = ''
                placeholder['poop_vali_ap'] = None
                placeholder['poop_test_ap'] = None
                placeholder['nocls_vali_ap'] = None
                placeholder['nocls_test_ap'] = None
                placeholder['status'] = 'no_finite_selection'
                selected_row = placeholder
            else:
                continue
        selected_rows.append(selected_row)

    deduped_rows = {}
    for row in selected_rows:
        key = (row['model_family'], row.get('config_tag', 'baseline'), row['subset_name'], row['train_size'])
        prev = deduped_rows.get(key)
        if prev is None or _row_preference_key(row) > _row_preference_key(prev):
            deduped_rows[key] = row
    selected_rows = list(deduped_rows.values())
    selected_rows = [_augment_with_run_metadata(row) for row in selected_rows]

    run_links_dpath = out_dpath / 'run_links'
    for row in selected_rows:
        link_fpath = run_links_dpath / _safe_link_name(row['model_family']) / _safe_link_name(row.get('config_tag', 'baseline')) / _safe_link_name(row['subset_name'])
        target = pathlib.Path(row['run_dpath'])
        if target.exists():
            _refresh_symlink(link_fpath, target)
            row['run_link_fpath'] = str(link_fpath)
        else:
            row['run_link_fpath'] = ''

    summary_fpath = out_dpath / 'benchmark_summary.tsv'
    fieldnames = [
        'model_family',
        'config_tag',
        'config_label',
        'status',
        'subset_name',
        'train_size',
        'train_batch_size',
        'vali_batch_size',
        'num_gpus_requested',
        'selected_candidate_id',
        'poop_vali_ap',
        'poop_test_ap',
        'nocls_vali_ap',
        'nocls_test_ap',
        'run_dpath',
        'run_link_fpath',
        'run_manifest_fpath',
        'summary_fpath',
    ]
    with summary_fpath.open('w', newline='') as file:
        writer = csv.DictWriter(file, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(selected_rows, key=lambda row: (row['model_family'], row.get('config_tag', 'baseline'), row['train_size'])):
            writer.writerow({key: row.get(key, '') for key in fieldnames})

    plots_dpath = out_dpath / 'plots'
    plots_dpath.mkdir(parents=True, exist_ok=True)
    # Keep legacy path for backward compat
    plot_fpath = out_dpath / 'train_size_curve.png'

    if selected_rows:
        _generate_plots(selected_rows, plots_dpath, plot_fpath)

    manifest = {
        'benchmark_root': str(benchmark_root),
        'summary_fpath': str(summary_fpath),
        'num_rows': len(selected_rows),
        'plots_dpath': str(plots_dpath),
        'plot_fpath': str(plot_fpath),
        'run_links_dpath': str(run_links_dpath),
    }
    (out_dpath / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DINO detector benchmark analysis complete')
    print(f'  BENCHMARK_ROOT         {benchmark_root}')
    print(f'  SUMMARY_FPATH          {summary_fpath}')
    print(f'  PLOTS_DPATH            {plots_dpath}')
    print(f'  RUN_LINKS_DPATH        {run_links_dpath}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
