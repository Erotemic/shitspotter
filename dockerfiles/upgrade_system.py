#!/usr/bin/env python3
"""
Update a Docker container by syncing specified repos from a staging directory
and committing the result as a new image.

This lets us update our containers without a complete docker rebuild.
"""
import time
import ubelt as ub
import kwutil
import tempfile
import scriptconfig as scfg


class UpdateContainerConfig(scfg.DataConfig):
    """
    Update Git repos in a container using local staging repos.
    """
    base_image = scfg.Value('image_name:latest', help='Base Docker image to start from', position=1)
    new_image = scfg.Value('image_name:updated', help='Name of the new image to commit', position=2)
    staging_dir = scfg.Value('.staging', help='Local directory with updated repo sources')
    repos = scfg.Value('.staging/repos.yml', help='YAML mapping of repo names to container paths')


def main(argv=True, **kwargs):
    config = UpdateContainerConfig.cli(argv=argv, data=kwargs)

    base_image = config.base_image
    new_image = config.new_image
    staging_dir = ub.Path(config.staging_dir).resolve()
    container_name = f'update-temp-container-{int(time.time())}'

    # Load repo → container path mapping
    repos = kwutil.Yaml.coerce(config.repos)
    assert isinstance(repos, dict), "Config file must contain a mapping of repo → container path"

    repo_items = repos['repos']

    # --- Start the container ---
    print(f"[INFO] Starting container from image: {base_image}")
    out = ub.cmd([
        'docker', 'run', '-d', '--rm',
        '--name', container_name,
        '--mount', f'type=bind,source={staging_dir},target=/mnt/staging,readonly',
        base_image, 'sleep', 'infinity'
    ], verbose=3)
    try:
        out.check_returncode()
    except Exception as ex:
        stderr = ex.stderr if ex.stderr else ''
        if 'already in use by container' in stderr or 'Conflict. The container name' in stderr:
            msg = ub.codeblock(
                f'''
                [ERROR] Container named '{container_name}' is already running.
                To stop the existing container, run:
                    docker stop {container_name}
                ''')
            ex = kwutil.util_exception.add_exception_note(ex, msg)
        raise ex

    update_script_parts = [ub.codeblock(
        """
        #!/bin/bash
        set -euo pipefail
        """)]
    for repo_item in repo_items:
        repo_name = repo_item['name']
        staging_path = f"/mnt/staging/{repo_name}"
        container_path = f"/root/code/{repo_name}"

        # TODO: update this as needed (maybe need to reinstall on update?)
        update_script_parts += [ub.codeblock(
            f"""
            echo '--- Updating {repo_name} ---'
            git config --global --add safe.directory '{staging_path}/.git'
            if [ -d '{container_path}/.git' ]; then
                cd '{container_path}'
                if git remote | grep -q '^staging$'; then
                    git remote remove staging
                fi
                git remote add staging '{staging_path}'
                git fetch staging
                default_branch=$(git remote show staging | grep 'HEAD branch' | cut -d: -f2 | tr -d ' ')
                git reset --hard staging/$default_branch
            else
                echo 'Repo not found at {container_path}, cloning from staging...'
                git clone '{staging_path}' '{container_path}'
                cd '{container_path}'
                uv pip install -e .
            fi
            """)]

    # Write script to temp file
    text = '\n'.join(update_script_parts)
    print(text)

    with tempfile.NamedTemporaryFile('w', delete=False, prefix='update_script_', suffix='.sh') as tf:
        tf.write(text)
        temp_script_path = ub.Path(tf.name)

    print(f"[INFO] Written update script to: {temp_script_path}")
    dest_script_path = "/root/update_repos.sh"

    # Copy script into container
    ub.cmd(['docker', 'cp', str(temp_script_path), f"{container_name}:{dest_script_path}"], verbose=3, check=True)

    # Run script inside container (no chmod needed)
    ub.cmd(['docker', 'exec', container_name, 'bash', '-l', dest_script_path], verbose=3, check=True)

    # Remove temp file
    temp_script_path.unlink()
    print(f"[INFO] Removed temporary update script {temp_script_path}")

    # --- Commit container as new image ---
    print(f"\n[INFO] Committing updated container as image: {new_image}")
    ub.cmd(['docker', 'commit', container_name, new_image], verbose=3, check=True)

    # --- Cleanup ---
    print(f"[INFO] Stopping container: {container_name}")
    ub.cmd(['docker', 'stop', container_name], verbose=3, check=True)

    print(f"\n✅ Update complete: {new_image}")


if __name__ == '__main__':
    main()
