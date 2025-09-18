"""
uv pip install roboflow
"""
import kwutil
from roboflow import Roboflow
import rich
import os
import ubelt as ub
robo_dpath = ub.Path('/data/joncrall/dvc-repos/roboflow-repos').ensuredir()
os.chdir(robo_dpath)

ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
# workspace = rf.workspace("jon-crall")
# dataset = project.version("1").download("yolov5")  # or c

projects = kwutil.Yaml.coerce(
    """
    # - project_id: pet-waste-detect-xvmnr Mine.
    # - project_id: dog-6eieq-5kx3q

    - workspace: "han-bjamu"
      project_id: "poppopo"
      version_id: "2"
      name: "poppopo Computer Vision Dataset"
      project_type: "semantic-segmentation"
      orig_url: https://universe.roboflow.com/han-bjamu/poppopo

    - workspace: ncue-n5bis
      project_id: dog-poop-c15x4-qolmq
      version_id: 1
      name: "Ｄｏｇ　ｐｏｏｐ"
      project_type: "semantic-segmentation"
      orig_url: https://universe.roboflow.com/ncue-n5bis/dog-poop-c15x4
      my_fork_id: jon-crall/dog-poop-c15x4-qolmq

    - workspace: project-oftnd
      project_id: dog-6eieq
      version_id: '2'
      id: project-oftnd/dog-6eieq/2
      project_type: "object-detection"
      orig_url: https://universe.roboflow.com/project-oftnd/dog-6eieq

    - workspace: benito
      project_id: dog-excrements
      version_id: '2'
      id: benito/dog-excrements/2
      project_type: "classification"
      url: https://universe.roboflow.com/benito/dog-excrements

    # - workspace: cogmodel
    #   project_id: poop_segmentation
    #   version_id: '2'
    #   my_fork_id: jon-crall/poop_segmentation-cxrst

    - workspace: cj-capstone
      project_id: dog-poop-13vwg
      version_id: '2'

    - workspace: pazini
      project_id: dog-poop-detection-uip1h
      version_id: '3'

    - workspace: dog-poop
      project_id: dog-poop-qdsbu
      version_id: '11'
      my_fork_id: jon-crall/dog-poop-qdsbu-mnmzh

    - workspace: dog-poop
      project_id: fake-dog-poop
      version_id: '4'

    - workspace: dog-poop
      project_id: fake-dog-poop-2
      version_id: '1'

    # - project_id: project-q89rt/stem-i/4
    #   notes: human-poop

    # - project_id: garrett-b/dog-poop-health-id/1

    - workspace: ncue-uhqpj
      project_id: dog-poop-mmvam
      my_fork_id: jon-crall/dog-poop-mmvam-8jiev
      version_id: '3'
      notes: "seems black and white for some reason. Manual cut/paste augmentation with speckle."

    """, backend='pyyaml')

for project_info in projects:
    print(f'project_info = {ub.urepr(project_info, nl=1)}')
    full_id = project_info.get('id')
    version = None
    full_id = None
    project = None
    dl_dpath = None
    workspace = None
    project_id = None
    version_id = None

    if full_id:
        parts = full_id.split('/')
        if len(parts) == 2:
            workspace, project_id = parts
        else:
            workspace, project_id, version_id = parts
    else:
        workspace = project_info.get('workspace')
        project_id = project_info['project_id']
        version_id = project_info.get('version_id', None)

    if version_id is not None:
        version_id = int(version_id)

    if 0:
        if version_id is None:
            # If we know the version id, we can look up some information
            ws = rf.workspace(workspace)
            project = ws.project(project_id)
            # project_info.update(project.__dict__)
            versions = project.versions()
            project_info['num_versions'] = len(versions)
            if len(versions):
                version = versions[0]
                full_id = version.id
                version_id = version.version

    if full_id is None:
        if all(x is not None for x in [workspace, project_id, version_id]):
            full_id = f'{workspace}/{project_id}/{version_id}'

    if full_id is not None:
        dl_dpath = (robo_dpath / full_id)

    project_info['full_id'] = full_id
    project_info['workspace'] = workspace
    project_info['project_id'] = project_id
    project_info['version_id'] = version_id
    project_info['dl_dpath'] = dl_dpath
    project_info['url'] = f'https://universe.roboflow.com/{workspace}/{project_id}'

    if 1:
        if full_id is not None:
            dl_dpath.ensuredir()
            os.chdir(dl_dpath)
            subdirs = [p for p in dl_dpath.ls() if p.is_dir()]

            # If the dataset has not been downloaded yet (weak check)
            if len(subdirs) == 0:
                if version is None:
                    # Ensure we have the version object
                    ws = rf.workspace(workspace)
                    project = ws.project(project_id)
                    version = project.version(version_id)

                # Call the download function
                if project.type == 'semantic-segmentation':
                    dataset = version.download('coco-segmentation')
                elif project.type == 'object-detection':
                    dataset = version.download('coco')
                elif project.type == 'classification':
                    dataset = version.download('clip')
                else:
                    dataset = version.download('coco')

                import xdev
                walker = xdev.DirectoryWalker(ub.Path(dataset.location)).build()
                walker.write_report(max_depth=0)

        candidates = list(dl_dpath.ls('*/*/_annotations.coco.json'))
        import kwcoco
        splits = {}
        for found in candidates:
            split_name = found.parent.name
            assert split_name in {'train', 'test', 'valid'}
            splits[split_name] = found
        project_info['splits'] = splits

projects = kwutil.Json.ensure_serializable(projects)
rich.print(kwutil.Yaml.dumps(projects))
# rich.print(f'projects = {ub.urepr(projects, nl=2)}')
os.chdir(robo_dpath)

rows = []
for project_info in projects:
    splits = project_info.get('splits')
    for split_name, fpath in splits.items():
        basic = kwcoco.CocoDataset(found).basic_stats()
        row = {
            'full_id': project_info['full_id'],
            'split': split_name,
            **basic
        }
        rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)
