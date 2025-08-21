"""
uv pip install roboflow
"""
import kwutil
from roboflow import Roboflow
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
    # - project_id: dog-poop-c15x4-qolmq

    - id: project-oftnd/dog-6eieq/2

    - id: benito/dog-excrements/2

    - id: jon-crall/poop_segmentation-cxrst
      orig_id: cogmodel/poop_segmentation

    - project_id: dog-poop-mmvam
      workspace: ncue-uhqpj
      my_fork_id: dog-poop-mmvam-8jiev

    - project_id: dog-poop-13vwg
      workspace: cj-capstone

    - workspace: pazini
      project_id: dog-poop-detection-uip1h

    - my_fork_id: dog-poop-qdsbu-mnmzh
      project_id: dog-poop-qdsbu
      workspace: dog-poop

    - project_id: fake-dog-poop
      workspace: dog-poop

    - project_id: fake-dog-poop-2
      workspace: dog-poop

    - project_id: project-q89rt/stem-i/4
      notes: human-poop

    - project_id: garrett-b/dog-poop-health-id/1

    """
)

for project_info in projects:
    full_id = project_info.get('id')
    if full_id:
        parts = full_id.split('/')
        if len(parts) == 2:
            workspace, project_id = parts
        else:
            workspace, project_id, version_id = parts
    else:
        workspace = project_info.get('workspace')
        project_id = project_info['project_id']
        version_id = None

    version = None
    full_id = None
    project = None

    if version_id is None:
        ws = rf.workspace(workspace)
        project = ws.project(project_id)
        project_info.update(project.__dict__)
        versions = project.versions()
        project_info['num_versions'] = len(versions)
        if len(versions):
            version = versions[0]
            full_id = version.id

    if full_id is not None:
        dl_dpath = (robo_dpath / full_id).ensuredir()
        os.chdir(dl_dpath)
        subdirs = [p for p in dl_dpath.ls() if p.is_dir()]
        if len(subdirs) == 0:
            if version is None:
                ws = rf.workspace(workspace)
                project = ws.project(project_id)
                version = project.version(version_id)
            if project.type == 'semantic-segmentation':
                version.download('coco-segmentation')
            else:
                version.download('coco')

print(f'projects = {ub.urepr(projects, nl=2)}')
os.chdir(robo_dpath)
