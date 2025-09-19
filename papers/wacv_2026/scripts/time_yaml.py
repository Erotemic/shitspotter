#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "ubelt",
#   "scriptconfig",
#   "rich",
#   "kwutil",
# ]
# ///
"""
References:
    https://chat.deepseek.com/a/chat/s/4993c78c-3f02-44ea-8b43-6fab9cb4753b

Usage:
    python ~/code/shitspotter/papers/wacv_2026/scripts/time_yaml.py \
        --before_script 'bash -c "
            ls -a
            ls
            ls -a
            ls"
        ' \
        --script 'python -c "if 1:
                import time
                for i in range(10):
                    time.sleep(0.1)
            "
        ' \
        --meta '
            action: test
            other: info
        '

"""
import uuid
import sys
import shlex
import subprocess
from datetime import datetime
import ubelt as ub
import kwutil
import rich
from rich.markup import escape
import scriptconfig as scfg


class TimeYamlCLI(scfg.DataConfig):
    script = scfg.Value('', str, help='param1', position=1)
    before_script = scfg.Value('', str, help='param1')
    meta = scfg.Value('', str, help='param1')

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from time_yaml import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = TimeYamlCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        time_command(config)

__cli__ = TimeYamlCLI


def time_command(config):
    meta = kwutil.Yaml.coerce(config.meta)
    if not meta:
        meta = {}
    assert isinstance(meta, dict)

    # Generate the UUID and timestamp first
    entry_uuid = str(uuid.uuid4())
    recording_time = str(ub.timestamp())

    # Prepare the time script (using bash time for formatting)
    time_cmd = f"/usr/bin/time -p {config.script.lstrip()}"

    if config.before_script:
        ub.cmd(config.before_script, verbose=3)

    result = {
        'uuid': entry_uuid,
        'recording_time': recording_time,
        **meta,
        # 'script': script,
    }

    try:
        # Run the script and capture output and timing
        start_time = datetime.now()

        out = ub.cmd(time_cmd, shell=True, verbose=3)

        end_time = datetime.now()
        result['time.start_date'] = str(start_time)

        # Parse the time output (comes from stderr)
        time_output = out.stderr
        time_data = {}
        for line in time_output.splitlines():
            if line.startswith('real'):
                time_data['real'] = line.split()[1]
            elif line.startswith('user'):
                time_data['user'] = line.split()[1]
            elif line.startswith('sys'):
                time_data['sys'] = line.split()[1]

        # Calculate duration in seconds
        duration = end_time.timestamp() - start_time.timestamp()

        # Format times to be more readable
        def format_time(seconds):
            seconds = float(seconds)
            minutes = int(seconds // 60)
            remaining = seconds % 60
            return f"{minutes}m{remaining:.3f}s"

        result['status'] = 'success'

        # Get end date
        end_date = datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')
        result['duration'] = f'{duration} seconds'
        result['time.real'] = format_time(time_data['real'])
        result['time.user'] = format_time(time_data['user'])
        result['time.sys'] = format_time(time_data['sys'])
        result['end_date'] = str(end_date)
    except subprocess.CalledProcessError as e:
        print(f'e={e}')
        result['status'] = 'failed'
        # If script fails, still record the attempt
        end_date = datetime.now().strftime('%a %b %d %I:%M:%S %p %Z %Y')
        result['end_date'] = str(end_date)

    out_lines = []
    out_lines.append(kwutil.Yaml.dumps(result).rstrip())
    full_command = shlex.join([sys.executable] + sys.argv)
    out_lines.append('script: |-')
    out_lines.append(ub.indent(full_command))
    # print(f'out_lines = {ub.urepr(out_lines, nl=1)}')
    out_text = '\n'.join(out_lines)
    # print(out_text)
    # print('---')
    print('  - ' + ub.indent(out_text).lstrip())


if __name__ == '__main__':
    __cli__.main()
