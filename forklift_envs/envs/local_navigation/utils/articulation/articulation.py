'''
Articulation implementation for detecting pallet collisions in a forklift.
'''

import re

from isaaclab.assets import Articulation
from isaacsim.core.utils.stage import get_current_stage
from pxr import PhysxSchema, Sdf, Usd, UsdGeom

class ForkliftArticulation(Articulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_contact_sensors()

    def prepare_contact_sensors(self):
        stage = get_current_stage()
        pattern = "/World/envs/env_.*/Forklift/.*(lift|body)$"
        matching_prims = []
        prim: Usd.Prim
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xform):
                prim_path: Sdf.Path = prim.GetPath()
                if re.match(pattern, prim_path.pathString):
                    matching_prims.append(prim_path)
        
        prims_per_env = 2
        for idx, prim in enumerate(matching_prims):
            env_idx = idx // prims_per_env
            contact_api: PhysxSchema.PhysxContactReportAPI = \
                PhysxSchema.PhysxContactReportAPI.Get(stage, prim)
            contact_api.CreateReportPairsRel().AddTarget(f"/World/envs/env_{env_idx}/Pallet")

# reference
class FrankaArticulation(Articulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_contact_sensors()

    def prepare_contact_sensors(self):
        stage = get_current_stage()
        pattern = "/World/envs/env_.*/Robot/.*(link1|link2|link3|link4|link5|link6|link7|hand)$"
        matching_prims = []
        prim: Usd.Prim
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xform):
                prim_path: Sdf.Path = prim.GetPath()
                if re.match(pattern, prim_path.pathString):
                    matching_prims.append(prim_path)

        prims_per_env = 8
        for idx, prim in enumerate(matching_prims):

            env_idx = idx // prims_per_env
            contact_api: PhysxSchema._physxSchema.PhysxContactReportAPI = \
                PhysxSchema._physxSchema.PhysxContactReportAPI.Get(stage, prim)
            contact_api.CreateReportPairsRel().AddTarget(f"/World/envs/env_{env_idx}/Table/Collisions/Cube")
