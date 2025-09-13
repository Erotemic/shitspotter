"""
SeeAlso:
    python ~/code/shitspotter/papers/neurips-2025/scripts/estimate_training_resources.py
    ~/code/shitspotter/papers/neurips-2025/scripts/build_v2_result_table.py
"""


def devcheck_main_stuff():
    import sys
    import ubelt as ub
    sys.path.append(str(ub.Path('~/code/shitspotter/papers/neurips-2025/scripts').expand()))
    from build_v2_result_table import load_aggregators
    from build_v2_result_table import process_aggregators
    from build_v2_result_table import report_resources
    aggregators_rows = load_aggregators()
    summary_df = process_aggregators(aggregators_rows)
    deduped = report_resources(aggregators_rows)

    mapping = {
        'heatmap_eval': 'vit',
        'extract_polygons': 'vit',
        'heatmap_pred': 'vit',
        'detectron_pred': 'mrcnn',
        'yolo_pred': 'yolo',
        'open_grounding_dino_pred': 'gdino',
        'grounding_dino_pred': 'gdino',
    }
    deduped['key'] = deduped['node_type'].apply(mapping.__getitem__)

    for k, group in deduped.groupby('key'):
        print(k)
        # print(group)
        print('  time', kwutil.timedelta.coerce(group['duration'].sum()).to('pint').to('days'))
        print('  kwh', group['kwh'].sum())
        print('  co2', group['co2_kg'].sum())
        ...



def main():
    """
    Notes:
        geowatch vit numbers filled from existing table.

        yolo9 is tracking all runs, which include botched ones done while getting everything working.
    """

    reg = kwutil.util_units.unit_registry()
    reg.define('co2 = []')
    reg.define('usd = []')
    reg.define('dollar = usd')
    reg.define('co2kg = co2 * kg')
    reg.define('kwh = 1000 * watt_hour')

    gpu_power = 345 * reg.watt

    real_co2 = 1.84 * reg.co2kg
    real_kwh = 8.76 * reg.kwh
    estimated_ratio = real_co2 / real_kwh
    print('emission_factor', estimated_ratio)

    dollar_per_kwh = 0.16 * reg.dollar / reg.kwh
    dollar_per_1000_kgco2 = 25 * reg.dollar / (1000 * reg.co2kg)

    # Fill in this with best estimates for final table
    data_table = [
        {'model': 'gwvit', 'phase': 'train', 'time': '158.95 days', 'energy': '1316.07 kwh', 'emissions': '276.37 co2kg', 'num_runs': 42},
        {'model': 'mrcnn', 'phase': 'train', 'time': '17.0 hours', 'energy': '5.917467994354167 kwh', 'emissions': '1.2426682788143752 co2kg', 'num_runs': ...},
        {'model': 'yolo9', 'phase': 'train', 'time': '99.41305537722221 hour', 'energy': ..., 'emissions': ...},
        {'model': 'gdino', 'phase': 'train', 'time': '7.76805556 hours', 'energy': ..., 'emissions': ..., 'num_runs': 1},

        {'model': 'gwvit', 'phase': 'test', 'time': '13.13 days', 'energy': '102.83 kwh', 'emissions': '21.6 co2kg'},
        {'model': 'mrcnn', 'phase': 'test', 'time': '0.5653703703703703 days', 'energy': '4.409049311843411 kwh', 'emissions': '0.927998761060484 co2kg'},
        {'model': 'yolo9', 'phase': 'test', 'time': '0.08109953703703703 days', 'energy': '0.1907299144027736 kwh', 'emissions': '0.022824279083487144 co2kg'},
        {'model': 'gdino', 'phase': 'test', 'time': '0.13366898148148146 days', 'energy': '0.294955172559941 kwh', 'emissions': '0.015369300793710499 co2kg'},
    ]

    for item in data_table:
        item['time'] = kwutil.timedelta.coerce(item['time']).to_pint()
        item['time'] = reg.Quantity(item['time'].m, item['time'].u)

        if isinstance(item['energy'], str):
            item['energy'] = reg.parse_expression(item['energy'])

        if isinstance(item['emissions'], str):
            item['emissions'] = reg.parse_expression(item['emissions'])

        if item['phase'] == 'train':
            if item['energy'] == ...:
                item['energy'] = item['time'] * gpu_power
            if item['emissions'] == ...:
                item['emissions'] = item['energy'] * estimated_ratio

        item['time'] = item['time'].to('days')
        item['energy'] = item['energy'].to('kwh')
        item['emissions'] = item['emissions'].to('co2kg')
        item.pop('num_runs', None)

        item['cost'] = ((item['emissions'] * dollar_per_1000_kgco2) + (item['energy'] * dollar_per_kwh)).to('dollar')


    itemized_df = pd.DataFrame(data_table)

    print(df)
    per_group = itemized_df.groupby('phase')[['time', 'energy', 'emissions', 'cost']].sum()
    total = itemized_df[['time', 'energy', 'emissions', 'cost']].sum()


    import copy
    human_table = copy.deepcopy(data_table)
    for item in per_group.reset_index().to_dict('records'):
        item['model'] = 'Total'
        human_table.append(item)
    item = total.to_dict()
    item['phase'] = 'Overall'
    item['model'] = 'Total'
    human_table.append(item)
    for item in human_table:
        t = round(item['time'].m, 2)
        e = round(item['energy'].m, 2)
        c = round(item['emissions'].m, 2)
        d = round(item['cost'].m, 2)
        item['time'] = f'{t:0.2f}'
        item['energy'] = f'{e:0.2f}'
        item['emissions'] = f'{c:0.2f}'
        item['cost'] = f'{d:0.2f}'

    df = pd.DataFrame(human_table)
    import sys
    import ubelt as ub
    sys.path.append(str(ub.Path('~/code/shitspotter/papers/neurips-2025/scripts').expand()))
    from build_v2_result_table import make_latex_table
    text = make_latex_table(df, index=False, bold_maxima=False)
    print(text)
