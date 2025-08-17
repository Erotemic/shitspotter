"""
python ~/code/shitspotter/papers/neurips-2025/scripts/estimate_training_resources.py
python ~/code/shitspotter/dev/poc/estimate_train_resources.py
"""
import kwutil.util_units
import ubelt as ub
import sys


def estimate_for_original_maskrcnn_and_vit():
    sys.path.append(ub.expandpath('~/code/shitspotter/dev/poc'))

    reg = kwutil.util_units.unit_registry()
    # gpu_power = 350 * reg.watt
    gpu_power = 345 * reg.watt
    time = 49.2 * reg.hour

    real_co2 = 1.84
    real_kwh = 8.76
    estimated_ratio = real_co2 / real_kwh
    print(estimated_ratio)

    co2kg_per_kwh = 0.210
    energy_usage = (gpu_power *  time).to(reg.kilowatt * reg.hour)

    co2_kg = energy_usage.m * co2kg_per_kwh
    print(f'{round(co2_kg, 1)} CO2 kg')

    dollar_per_kg = 0.015

    cost_to_offset = dollar_per_kg * co2_kg
    print(f'cost_to_offset = ${cost_to_offset:4.2f}')

    # Detectron training results
    runs_dpath = ub.Path('$HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs').expand()
    detectron_dpaths = [
        runs_dpath / 'train_baseline_maskrcnn_scratch_v4',
        runs_dpath / 'train_baseline_maskrcnn_v3',
    ]
    from estimate_train_resources import estimate_training_duration, find_offset_cost  # NOQA

    rows = []
    for train_dpath in detectron_dpaths:
        for dpath in train_dpath.ls():
            if dpath.is_dir():
                checkpoint_paths = list(dpath.glob('*.pth'))
                info = estimate_training_duration(checkpoint_paths)
                info['duration_human'] = kwutil.timedelta.coerce(
                    info['duration']).format(unit='auto', precision=2)
                info['num_checkpoints'] = len(checkpoint_paths)
                info['dpath'] = dpath
                info.update(find_offset_cost(info['duration']))
                rows.append(info)

    print(f'rows = {ub.urepr(rows, nl=2)}')

    r"""

    GeoWatch:

    train$^{*}$ & time        & 158.95 days
    train$^{*}$ & electricity & 1,316.07 kWh
    train$^{*}$ & emissions   & 276.37 \cotwo kg

    Detectron

    17.0 hours
    1.2426682788143752 CO2
    5.917467994354167 kWh

    total:
    159.66 days
    1321.99 kWh
    277.612 \cotwo kg
    """


def estimate_for_recent_dino_and_yolo():
    import kwutil
    import pandas as pd
    runs_dpath = ub.Path('$HOME/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs').expand()

    paths = [
        runs_dpath / 'mit-yolo-v04',
        runs_dpath / 'mit-yolo-v01',
        runs_dpath / 'shitspotter-yolo-train_imgs5747_1e73d54f-v5',
        runs_dpath / 'mit-yolo-v03',
    ]

    lightning_log_dpaths = []
    for dpath in paths:
        lightning_log_dpaths.extend(list((dpath / 'train').glob('*/lightning_logs')))
        ...

    rows = []
    for dpath in lightning_log_dpaths:
        for version_dpath in dpath.ls('version_*'):
            checkpoint_dpath = version_dpath / 'checkpoints'
            hparams_fpath = version_dpath / 'hparams.yaml'
            if checkpoint_dpath.exists():
                last_time = max([p.stat().st_mtime for p in checkpoint_dpath.ls('*.ckpt')])
                start_time = hparams_fpath.stat().st_mtime
                duration = last_time - start_time
                duration_h = kwutil.timedelta.coerce(duration).to_pint().to('hour')
                expt_name = dpath.relative_to(runs_dpath).parts[0]
                rows.append({
                    'version_dpath': version_dpath,
                    'expt_name': expt_name,
                    'duration_h': duration_h,
                })
    df = pd.DataFrame(rows)

    main_yolo_train = df[df['expt_name'] == 'shitspotter-yolo-train_imgs5747_1e73d54f-v5']
    main_yolo_duration = main_yolo_train['duration_h'].iloc[0]
    total_yolo_duration = df['duration_h'].sum()

    import xdev
    walker = xdev.DirectoryWalker(runs_dpath / 'grounding-dino-tune-v001/').build()
    walker.write_report()

    dino_path = runs_dpath / 'grounding-dino-tune-v001/'

    rows = []
    for p in dino_path.glob('*'):
        time = kwutil.datetime.coerce(p.stat().st_mtime)
        rows.append({
            'name': p.name,
            'time': time,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('time')
    df['time'].min()
    log_data = [kwutil.Json.coerce(line) for line in (dino_path / 'log.txt').read_text().strip().split('\n')]
    total_time = sum([kwutil.timedelta.coerce(row['epoch_time']) for row in log_data], start=kwutil.timedelta.coerce(0))
    dino_total_hours = kwutil.timedelta.coerce(total_time).to_pint().to('hours')
    print(f'dino_total_hours = {ub.urepr(dino_total_hours, nl=1)}')
    print(f'main_yolo_duration={main_yolo_duration}')
    print(f'total_yolo_duration={total_yolo_duration}')

    """

    Previously noted maskrcnn+vit training was 159.66 days

    So with YOLO and DINO we add

    from kwutil import util_units
    ureg = util_units.unit_registry()

    ((7.768055555555556 * ureg.hours) + (99.41305537722221 * ureg.hours) + (159.66 * ureg.days))

    Results in 3939.02111, hours
    Or 164.12588 days.

    """

    reg = kwutil.util_units.unit_registry()
    # gpu_power = 350 * reg.watt
    gpu_power = 345 * reg.watt
    time = 3939.02111 * reg.hour

    real_co2 = 1.84
    real_kwh = 8.76
    estimated_ratio = real_co2 / real_kwh
    print('emission_factor', estimated_ratio)

    co2kg_per_kwh = 0.210
    energy_usage = (gpu_power *  time).to(reg.kilowatt * reg.hour)
    print(f'energy_usage = {ub.urepr(energy_usage, nl=1)}')

    co2_kg = energy_usage.m * co2kg_per_kwh
    print(f'{round(co2_kg, 1)} CO2 kg')

    dollar_per_kwh = 0.16
    dollar_per_1000_kgco2 = 25

    energy_cost = energy_usage.m * dollar_per_kwh
    print(f'energy_cost={energy_cost}')
    co2_cost = co2_kg * dollar_per_1000_kgco2 / 1000
    print(f'co2_cost={co2_cost}')
    total_train_cost = energy_cost + co2_cost
    print(f'total_train_cost={total_train_cost}')

    r"""
    Gives us final total training numbers:

    emission_factor 0.21004566210045664
    energy_usage = 1358.9622829500001 kilowatt hour
    285.4 CO2 kg
    energy_cost=217.43396527200002
    co2_cost=7.1345519854875015
    total_train_cost=224.5685172574875

    total:
    164.13 days
    1358.96 kWh
    285.4 \\cotwo kg

    ### add in evaluation as well.

    # Original evaluation numbers
    15.6 days, consuming 109.63 kWh and emitting 23.0 co2

    # These are the total costs for DINO(z+t) + YOLO(p+s)
    total_new_kwh=0.4856850869627147
    total_new_time=5.154444444444444 hour
    total_new_co2=0.038193579877197646
    ---
    new_total_eval_time=15.81 day
    new_total_eval_kwh=110.12
    new_total_eval_co2=23.04

    """

    new_total_eval_time = ((15.6 * reg.days) + (5.154444444444444 * reg.hours)).to('days')
    new_total_eval_kwh = 109.63 + 0.4856850869627147
    new_total_eval_co2 = 23.0 + 0.038193579877197646
    print(f'new_total_eval_time={new_total_eval_time}')
    print(f'new_total_eval_kwh={new_total_eval_kwh}')
    print(f'new_total_eval_co2={new_total_eval_co2}')

    ###
    main_expt_total_energy = 110.12 + 1358.96
    main_expt_total_co2 = 23.0 + 285.4
    dollar_per_kwh = 0.16
    dollar_per_1000_kgco2 = 25

    energy_cost = main_expt_total_energy * dollar_per_kwh
    print(f'energy_cost={energy_cost}')
    co2_cost = main_expt_total_co2 * dollar_per_1000_kgco2 / 1000
    print(f'co2_cost={co2_cost}')
    total_cost = energy_cost + co2_cost
    print(f'total_cost={total_cost}')
