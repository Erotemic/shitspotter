"""
CommandLine:
    python ~/code/shitspotter/papers/application-2024/scripts/gather_measurements.py

SeeAlso:
    ~/code/shitspotter/papers/application-2024/transfer_measurements.yaml
"""
import kwutil
import rich
import pandas as pd
import ubelt as ub


def main():
    fpath = ub.Path('~/code/shitspotter/papers/application-2024/transfer_measurements.yaml').expand()
    text = fpath.read_text()
    data = kwutil.Yaml.loads(text)

    rows = data['network_measurements']
    for r in rows:
        try:
            if r['src_machine'] == r['dst_machine']:
                r['method'] = 'rsync-local'
        except KeyError:
            ...
        try:
            if r['src_machine'] == r['dst_machine']:
                r['method'] = 'rsync-local'
        except KeyError:
            ...
    subrows = [r for r in rows if r['action'] in {'transfer', 'upload'}]
    # table = pd.DataFrame(rows)
    subtable = pd.DataFrame(subrows)

    rows_of_interest = [
        'recording_time',
        'uuid',
        'method',
        'action',
        'duration',
        'time.start_date',
        'time.end_date',
        'src_machine',
        'dst_machine',
        # 'history.downloading_time',
        'status',
    ]

    subtable2 = subtable[rows_of_interest]

    def normalize_timedelta(data):
        return kwutil.timedelta.coerce(data, nan_policy='return-nan', none_policy='return-nan')

    def normalize_datetime(data):
        return kwutil.datetime.coerce(data, nan_policy='return-nan', none_policy='return-nan')

    subtable2['duration'] = subtable2['duration'].apply(normalize_timedelta)
    subtable2['recording_time'] = subtable2['recording_time'].apply(normalize_datetime)
    subtable2['uuid'] = subtable2['uuid'].apply(lambda x: str(x)[0:8])
    subtable2['time.start_date'] = subtable2['time.start_date'].apply(normalize_datetime)
    subtable2['time.end_date'] = subtable2['time.end_date'].apply(normalize_datetime)
    rich.print(subtable2.to_string())

    subtable3 = subtable2[subtable2['status'] != 'outlier']
    groups = subtable3.groupby('method')
    for method, group in groups:
        rich.print('-----')
        rich.print(method)
        rich.print('-----')
        rich.print(group)

    stats = groups['duration'].describe()

    glance_table = stats[['count', 'mean', 'std', 'min', 'max']]

    display_table = glance_table.copy()

    for c in display_table.columns[1:]:
        display_table[c] = display_table[c].apply(lambda x: '{:0.2f}'.format((x.total_seconds() / (60 * 60))))

    rich.print(display_table)

    total_time = subtable2['duration'].sum()
    print(total_time)

    latex_text = display_table.style.to_latex()
    print(latex_text)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/papers/application-2024/scripts/gather_measurements.py
    """
    main()
