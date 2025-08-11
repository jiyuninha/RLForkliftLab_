import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf

PALLET_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pallets", "Pallet_A1.usd")
    
@configclass
class PalletSceneCfg(InteractiveSceneCfg):
    """
    Pallet Scene Configurations
    """
    # ground = AssetBaseCfg(
    #     prim_path="/World/defaultGroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg()
    # )

    pallet_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pallet",
        spawn=sim_utils.UsdFileCfg(usd_path=PALLET_USD_PATH, 
              scale=[0.01, 0.01, 0.01],
              ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, 0.0, 0.0)),
    )

    target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-2.0, 0.0, 0.0),
        ),
    )

    