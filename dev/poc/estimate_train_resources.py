#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class EstimateTrainResourcesCLI(scfg.DataConfig):
    run_dpath = '/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/'
    # param1 = scfg.Value(None, help='param1')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.expandpath('~/code/shitspotter/dev/poc'))
            >>> from estimate_train_resources import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = EstimateTrainResourcesCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        run_dpath = ub.Path(config['run_dpath'])
        version_dpaths = list(run_dpath.glob('*/lightning_logs/version_*'))

        import kwutil
        rows = []
        for dpath in version_dpaths:
            expt_name = dpath.parent.parent.name
            dpath = ub.Path(dpath)
            ckpt_dpath = dpath / 'checkpoints'
            if not ckpt_dpath.exists():
                continue

            hparams_fpath = (dpath / 'hparams.yaml')
            if not hparams_fpath.exists():
                continue

            checkpoint_fpaths = list(ckpt_dpath.glob('*.ckpt'))
            print(f'checkpoint_fpaths = {ub.urepr(checkpoint_fpaths, nl=1)}')

            hparams_time = kwutil.datetime.coerce(hparams_fpath.stat().st_mtime)

            ckpt_times = []
            for cpkt_fpath in checkpoint_fpaths:
                ckpt_time = kwutil.datetime.coerce(cpkt_fpath.stat().st_mtime)
                ckpt_times.append(ckpt_time)

            min_dtime = min(hparams_time, *ckpt_times)
            max_dtime = max(hparams_time, *ckpt_times)
            duration = max_dtime - min_dtime
            row = {
                'expt_name': expt_name,
                'dpath': dpath,
                'duration': duration,
            }
            rows.append(row)

        # Infer more information from each training directory
        # including lineage
        import kwutil
        for row in ub.ProgIter(rows, desc='Loading more train info'):
            dpath = row['dpath']

            telemetry_fpath = dpath / 'telemetry.json'
            if telemetry_fpath.exists():
                telemetry_data = kwutil.Yaml.coerce(telemetry_fpath)
                telemetry_data['properties']['start_timestamp']
                telemetry_data['properties']['stop_timestamp']
                # emissions data? Not recorded for training yet :(

            config_fpath = dpath / 'config.yaml'
            hparams_fpath = dpath / 'hparams.yaml'
            train_config = kwutil.Yaml.coerce(config_fpath)
            init = train_config['initializer']['init']
            row['init'] = init
            parent_expt_name = None
            if init != 'noop':
                init_fpath = ub.Path(init)
                if init_fpath.parent.name == 'models':
                    parent_expt_name = init_fpath.name.split('-epoch')[0].split('_epoch')[0]
                elif init_fpath.parent.parent.name == 'packages':
                    parent_expt_name = init_fpath.parent.name
                elif init_fpath.parent.name == 'checkpoints':
                    parent_expt_name = init_fpath.parent.parent.parent.parent.name
                else:
                    parent_expt_name = None
                    ...
            row['parent_expt_name'] = parent_expt_name

        import pandas as pd
        data = pd.DataFrame(rows)
        rich.print(data.groupby('expt_name')['duration'].sum())

        for expt_name, group in data.groupby('expt_name'):
            parent_names = group['parent_expt_name'].unique()
            print(f'expt_name={expt_name} -> parent_names={parent_names}')

        # import kwutil.util_units
        # reg = kwutil.util_units.unit_registry()
        # gpu_power = 350 * reg.watt
        # time = 49.2 * reg.hour
        # co2kg_per_kwh = 0.210
        # energy_usage = (gpu_power *  time).to(reg.kilowatt * reg.hour)
        # co2_kg = energy_usage.m * co2kg_per_kwh
        # print(f'{round(co2_kg, 1)} CO2 kg')
        # dollar_per_kg = 0.015
        # cost_to_offset = dollar_per_kg * co2_kg
        # print(f'cost_to_offset = ${cost_to_offset:4.2f}')

        def find_offset_cost(total_delta):
            import kwutil.util_units
            reg = kwutil.util_units.unit_registry()
            gpu_power = 350 * reg.watt
            num_hours = total_delta.total_seconds() / (60 * 60)
            time = num_hours * reg.hour
            co2kg_per_kwh = 0.210
            energy_usage = (gpu_power *  time).to(reg.kilowatt * reg.hour)
            print('kwh: ', energy_usage)
            co2_kg = energy_usage.m * co2kg_per_kwh
            print(f'{round(co2_kg, 1)} CO2 kg')
            dollar_per_kg = 0.015
            cost_to_offset = dollar_per_kg * co2_kg
            print(f'cost_to_offset = ${cost_to_offset:4.2f}')

        all_durations = data.groupby('expt_name')['duration'].sum()
        # paper_models = subdata[subdata['expt_name'].str.contains('noboxes')]
        # paper_models.groupby('expt_name')['duration'].sum()
        num_expts = len(all_durations)
        print('all sums')
        print(f'num_expts={num_expts}')
        total_gpu_hours = all_durations.sum()

        import kwutil
        import kwutil.util_units
        ureg = kwutil.util_units.unit_registry()
        print((total_gpu_hours.total_seconds() * ureg.seconds).to(ureg.day))
        print((all_durations.mean().total_seconds() * ureg.seconds).to(ureg.day))
        find_offset_cost(total_gpu_hours)

        if True:
            # HACK
            subdata = data[data.expt_name.str.contains('noboxes')]
            subdurations = subdata.groupby('expt_name')['duration'].sum()
            num_sub_expts = len(subdurations)
            # paper_models = subdata[subdata['expt_name'].str.contains('noboxes')]
            # paper_models.groupby('expt_name')['duration'].sum()
            print('noboxes sums')
            print(f'num_sub_expts={num_sub_expts}')
            total_gpu_hours = subdurations.sum()
            print(total_gpu_hours)
            print(subdurations.mean())
            print((total_gpu_hours.total_seconds() * ureg.seconds).to(ureg.day))
            print((subdurations.mean().total_seconds() * ureg.seconds).to(ureg.day))
            find_offset_cost(total_gpu_hours)


__cli__ = EstimateTrainResourcesCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/dev/poc/estimate_train_resources.py
        python -m estimate_train_resources
    """
    __cli__.main()
