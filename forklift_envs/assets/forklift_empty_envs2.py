# Copyright (c) 2025-2027, Inha University (SPARO Lab)
# Author: Jiyun Lee
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Collect Training Data
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Creating a forklift base environment")
# parser.add_argument(
#     "--scene", default="warehouse", choices=["warehouse", "warehouse_multiple_shelves", "full_warehouse"], type=str, help="Scene to load."
# )
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
# parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch
import re

import omni
from pxr import PhysxSchema, Sdf, Usd, UsdGeom


# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab.terrains import TerrainImporterCfg
# import isaaclab.envs.mdp as mdp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import actuator_pd, DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions
from isaacsim.core.utils.stage import get_current_stage

##
# Pre-defined configs
##

# FORKLIFT_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Robots/Forklift/forklift_b.usd"
FORKLIFT_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "forklift_b", "forklift_b.usd")

PALLET_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "pallets", "Pallet_A1.usd")

##
# Available strings: ['lift_joint', 'back_wheel_swivel', 'front_right_roller',
# 'back_left_roller', 'back_right_roller', 'front_left_roller', 'back_wheel_drive']
##

WHEEL_JOINTS = ['back_wheel_swivel', 'back_wheel_drive', 'front_right_roller', 'back_left_roller', 'back_right_roller', 'front_left_roller'] 
LIFT_JOINT = ["lift_joint"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ForkliftWheelActuator(Articulation):
    """Actuator for Forklift wheels"""
    def __init__(self, num_envs: int):
        #self.joint_names_expr = "|".join(WHEEL_JOINTS)
        # actuator_cfg = ImplicitActuatorCfg(
        #     class_type=actuator_pd.DCMotor,
        #     joint_names_expr=self.joint_names_expr,
        #     effort_limit=100.0,
        #     velocity_limit=10.0,
        #     stiffness=0.0,
        #     damping=0.0,
        #     saturation_effort=50.0
        # )
        # self.cfg = actuator_cfg
        super().__init__(cfg=actuator_cfg, num_envs=num_envs, device=DEVICE)
        self.prepare_contact_sensors()

    def prepare_contact_sensors(self):
        stage = get_current_stage()
        pattern = "/World/envs/env_.*/Forklift/.join(WHEEL_JOINTS)$"
        matching_prims = []
        prim: Usd.Prim
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Xform):
                prim_path: Sdf.Path = prim.GetPath()
                if re.match(pattern, prim_path.pathString):
                    matching_prims.append(prim_path)

        for prim in matching_prims:
            contact_api: PhysxSchema.PhysxContactReportAPI = \
                PhysxSchema.PhysxContactReportAPI.Get(stage, prim)
            contact_api.CreateReportPairsRel().AddTarget("/World/envs/env_.*/Pallet")

    # def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
    #     self.joint_names_expr = "|".join(WHEEL_JOINTS)
    #     return ArticulationActions(
    #         joint_effort=torch.clamp(control_action.joint_effort, -self.cfg.effort_limit, self.cfg.effort_limit),
    #         joint_vel=torch.clamp(control_action.joint_vel, -self.cfg.velocity_limit, self.cfg.velocity_limit),
    #         joint_pos=joint_pos,
    #     )

class ForkliftLiftActuator(Articulation):
    """Actuator for Forklift lifting mechanism"""
    def __init__(self, num_envs: int):
        actuator_cfg = ImplicitActuatorCfg(
            class_type=actuator_pd.IdealPDActuator,
            joint_names_expr="|".join(LIFT_JOINT),
            effort_limit=200.0,
            velocity_limit=5.0,
            stiffness=100.0,
            damping=10.0
        )
        super().__init__(cfg=actuator_cfg, num_envs=num_envs, device=DEVICE)

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        return ArticulationActions(
            joint_effort=torch.clamp(control_action.joint_effort, -self.cfg.effort_limit, self.cfg.effort_limit),
            joint_vel=torch.clamp(control_action.joint_vel, -self.cfg.velocity_limit, self.cfg.velocity_limit),
            joint_pos=joint_pos,
        )

@configclass
class ForkliftSceneCfg(InteractiveSceneCfg):
    """Configuration for a simple scene with a Forklift robot."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    forklift = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Forklift",
        spawn=sim_utils.UsdFileCfg(usd_path=FORKLIFT_USD_PATH),
        actuators={
            "wheel_actuator": ImplicitActuatorCfg(
                #class_type=ImplicitActuatorCfg,
                joint_names_expr="|".join(WHEEL_JOINTS),
                effort_limit=100.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "lift_actuator": ImplicitActuatorCfg(
                #class_type=ImplicitActuatorCfg,
                joint_names_expr="|".join(LIFT_JOINT),
                effort_limit=200.0,
                velocity_limit=5.0,
                stiffness=100.0,
                damping=10.0
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(-2.0, 0.0, 0.0))
    )

    pallet = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pallet",
        spawn=sim_utils.UsdFileCfg(usd_path=PALLET_USD_PATH, scale=[0.01, 0.01, 0.01]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(2.0,0.0,0.0)),
        debug_vis=True, # 충돌 시 해당 팔레트에 대해 디버그 시각화가 활성화
    )

##
# Main function
##

def main():
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Load scene config
    scene_cfg = ForkliftSceneCfg(num_envs=args_cli.num_envs, env_spacing=10)
    print("[INFO]: Scene is being loaded...")

    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Scene successfully loaded!")

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete!!!")

    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()