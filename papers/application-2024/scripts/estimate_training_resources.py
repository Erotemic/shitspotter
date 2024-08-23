"""
python ~/code/shitspotter/papers/application-2024/scripts/estimate_training_resources.py
python ~/code/shitspotter/dev/poc/estimate_train_resources.py
"""
import kwutil.util_units

reg = kwutil.util_units.unit_registry()
# gpu_power = 350 * reg.watt
gpu_power = 345 * reg.watt
time = 49.2 * reg.hour

co2kg_per_kwh = 0.210
energy_usage = (gpu_power *  time).to(reg.kilowatt * reg.hour)

co2_kg = energy_usage.m * co2kg_per_kwh
print(f'{round(co2_kg, 1)} CO2 kg')

dollar_per_kg = 0.015

cost_to_offset = dollar_per_kg * co2_kg
print(f'cost_to_offset = ${cost_to_offset:4.2f}')
