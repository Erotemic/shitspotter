#!/usr/bin/env python3
"""
Aggregate benchmark summaries and draw train-size curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib


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


def main(argv=1) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv if isinstance(argv, list) else None)
    benchmark_root = pathlib.Path(args.benchmark_root)
    out_dpath = pathlib.Path(args.out_dpath)
    out_dpath.mkdir(parents=True, exist_ok=True)

    rows = []
    for summary_fpath in sorted(benchmark_root.glob('runs/*/*/summary.tsv')):
        model_family = summary_fpath.parents[2].name
        subset_name = summary_fpath.parents[1].name
        for row in _read_summary_rows(summary_fpath):
            row = dict(row)
            row['model_family'] = model_family
            row['subset_name'] = subset_name
            row['train_size'] = int(row['train_size'])
            row['poop_vali_ap'] = _coerce_float(row.get('poop_vali_ap'))
            row['poop_test_ap'] = _coerce_float(row.get('poop_test_ap'))
            row['nocls_vali_ap'] = _coerce_float(row.get('nocls_vali_ap'))
            row['nocls_test_ap'] = _coerce_float(row.get('nocls_test_ap'))
            rows.append(row)

    summary_fpath = out_dpath / 'benchmark_summary.tsv'
    fieldnames = [
        'model_family',
        'subset_name',
        'train_size',
        'selected_candidate_id',
        'poop_vali_ap',
        'poop_test_ap',
        'nocls_vali_ap',
        'nocls_test_ap',
        'run_dpath',
        'summary_fpath',
    ]
    with summary_fpath.open('w', newline='') as file:
        writer = csv.DictWriter(file, delimiter='\t', fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})

    if rows:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        families = sorted({row['model_family'] for row in rows})
        for family in families:
            family_rows = sorted(
                [row for row in rows if row['model_family'] == family and row['poop_vali_ap'] is not None],
                key=lambda row: row['train_size'],
            )
            if not family_rows:
                continue
            ax.plot(
                [row['train_size'] for row in family_rows],
                [row['poop_vali_ap'] for row in family_rows],
                marker='o',
                label=f'{family} vali',
            )
            test_rows = [row for row in family_rows if row['poop_test_ap'] is not None]
            if test_rows:
                ax.plot(
                    [row['train_size'] for row in test_rows],
                    [row['poop_test_ap'] for row in test_rows],
                    marker='s',
                    linestyle='--',
                    label=f'{family} test',
                )
        ax.set_xlabel('Training size')
        ax.set_ylabel('Box AP (poop)')
        ax.set_title('DINO detector benchmark')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dpath / 'train_size_curve.png', dpi=200)

    manifest = {
        'benchmark_root': str(benchmark_root),
        'summary_fpath': str(summary_fpath),
        'num_rows': len(rows),
        'plot_fpath': str(out_dpath / 'train_size_curve.png'),
    }
    (out_dpath / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
