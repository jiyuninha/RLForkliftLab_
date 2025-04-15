# Copyright (c) 2025-2027, Inha University (SPARO Lab)
# Author: Jiyun Lee
# All rights reserved.
# Final Environment
# SPDX-License-Identifier: BSD-3-Clause

"""
Final Environment
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

STEER_JOINT = ['back_wheel_swivel']
DRIVE_JOINT = ['back_wheel_drive']
PASSIVE_JOINTS = ['front_right_roller', 'back_left_roller', 'back_right_roller', 'front_left_roller'] 
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
        #super().__init__(cfg=actuator_cfg, num_envs=num_envs, device=DEVICE)
        self.prepare_contact_sensors()

    def prepare_contact_sensors(self):
        print("[DEBUG] prepare_contact_sensor")
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

# (여기서부터)
# 랜덤 쿼터니언 생성 (정원혁)

from pxr import PhysxSchema, Sdf, Usd, UsdGeom, Gf
import random, math

def quat_from_yaw():
    y = random.uniform(-math.pi/2, math.pi/2) # -90 ~ 90 degree
    w = math.cos(y / 2)
    z = math.sin(y / 2)
    return Gf.Quatd(w, Gf.Vec3d(0, 0, z))

# 무작위 회전 (정원혁)
def randomize_orientation(prim_path_expr: str):
    """Randomize the color of the geometry."""
    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            orient_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:orient")
            if orient_spec is None:
                orient_spec = Sdf.AttributeSpec(prim_spec, "xformOp:orient", Sdf.ValueTypeNames.Quatd)
            orient_spec.default = quat_from_yaw()
# (여기까지 수정함)

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
            "steer_actuator": ImplicitActuatorCfg(
                #class_type=ImplicitActuatorCfg,
                joint_names_expr="|".join(STEER_JOINT),
                effort_limit=20.0,
                velocity_limit=3.0,
                stiffness=100.0,
                damping=1000.0,
            ),
            "drive_actuator": ImplicitActuatorCfg(
                joint_names_expr="|".join(DRIVE_JOINT),
                velocity_limit=6,
                effort_limit=12,
                stiffness=100.0,
                damping=4000.0,
            ),
            "passive_actuator": ImplicitActuatorCfg(
                joint_names_expr="|".join(PASSIVE_JOINTS),
                effort_limit=0,
                velocity_limit=12.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "lift_actuator": ImplicitActuatorCfg(
                #class_type=ImplicitActuatorCfg,
                joint_names_expr="|".join(LIFT_JOINT),
                effort_limit=200.0,
                velocity_limit=5.0,
                stiffness=8000.0,
                damping=1000.0
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(pos=(4.0, 0.0, 0.0))
    )

    pallet = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pallet",
        spawn=sim_utils.UsdFileCfg(usd_path=PALLET_USD_PATH, scale=[0.01, 0.01, 0.01]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0,0.0,0.0)), # axis-x rotation 90 degree ,rot=(0.7071, 0.0, 0.0, 0.7071)
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

# (여기서부터)    
    # Randomizing..
    randomize_orientation(scene_cfg.pallet.prim_path)
# (여기까지 수정함)
    print("[INFO]: Scene successfully loaded!")

    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete!!!")

    forklift: Articulation = scene["forklift"]

    drive_joint_index = forklift.find_joints(DRIVE_JOINT) # drive joint만 찾기
    steer_joint_index = forklift.find_joints(STEER_JOINT)

    while simulation_app.is_running():
        velocities = torch.torch.zeros_like(forklift.data.default_joint_vel)
        velocities[:, drive_joint_index[0]] = -0.5
        velocities[:, steer_joint_index[0]] = 0.1

        forklift.set_joint_velocity_target(velocities)
        scene.write_data_to_sim()
        sim.step()

if __name__ == "__main__":
    main()
    simulation_app.close()