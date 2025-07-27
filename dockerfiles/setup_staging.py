#!/usr/bin/env python3
"""
Create or update the staging directory with clean versions of the repos that we
can use to build our docker images.

Run this script from the dockerfiles directory:
    cd ~/code/shitspotter
    python ./dockerfiles/setup_staging.py
"""

import os
import subprocess
import yaml
from pathlib import Path
import sys


def run_cmd(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def parse_remote_url(remote_url):
    """Handle optional @branch suffix in remote_url"""
    if '@' in remote_url and not remote_url.startswith('git@'):
        url, branch = remote_url.rsplit('@', 1)
    else:
        url = remote_url
        branch = None
    return url, branch


def main():
    # Load repos.yaml

    # FIXME: better path
    yaml_path = Path(__file__).parent / 'repos.yaml'
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    repos = config['repos']

    # Find shitspotter root
    # FIXME: bad checks
    shitspotter_repo = next((r for r in repos if r['name'] == 'shitspotter'), None)
    if not shitspotter_repo:
        print("ERROR: No repo named 'shitspotter' found in repos.yaml", file=sys.stderr)
        sys.exit(1)

    shitspotter_root = Path(os.path.expanduser(shitspotter_repo['local_path']))
    if not (shitspotter_root / '.git').is_dir():
        print(f"ERROR: {shitspotter_root} is not a valid git repository", file=sys.stderr)
        sys.exit(1)

    staging_dir = shitspotter_root / '.staging'
    staging_dir.mkdir(exist_ok=True)
    print(f"Updating staging repos in: {staging_dir}")

    for repo in repos:
        name = repo['name']
        local_path = Path(os.path.expanduser(repo['local_path']))
        remote_url, url_branch = parse_remote_url(repo['remote_url'])

        if not (local_path / '.git').is_dir():
            print(f"ERROR: {local_path} is not a git repository", file=sys.stderr)
            sys.exit(2)

        print(f"Processing '{name}' from {local_path}")

        # Get current branch and commit of the local repo
        current_branch = run_cmd("git rev-parse --abbrev-ref HEAD", cwd=local_path)
        current_commit = run_cmd("git rev-parse HEAD", cwd=local_path)

        if url_branch:
            branch = url_branch
        else:
            branch = current_branch

        staging_path = staging_dir / name

        if (staging_path / '.git').is_dir():
            print(f" Repo '{name}' already exists in staging. Updating...")
            run_cmd("git fetch", cwd=staging_path)
            run_cmd(f"git checkout {branch}", cwd=staging_path)
            run_cmd(f"git reset --hard {current_commit}", cwd=staging_path)
        else:
            print(f" Cloning repo '{name}' into staging...")
            run_cmd(f"git clone --branch {branch} {local_path} {staging_path}")
            run_cmd(f"git reset --hard {current_commit}", cwd=staging_path)

        print(f" '{name}' updated to branch '{branch}' at commit '{current_commit}'")

    print("Staging complete.")

if __name__ == '__main__':
    main()
