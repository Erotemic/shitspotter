"""
python ~/code/shitspotter/papers/neurips-2025/scripts/estimate_training_resources.py
python ~/code/shitspotter/dev/poc/estimate_train_resources.py
"""
import kwutil.util_units
import ubelt as ub
import sys
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

"""


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
