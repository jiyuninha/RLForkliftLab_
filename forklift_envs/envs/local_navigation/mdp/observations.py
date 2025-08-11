from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.assets import AssetBase, RigidObject
from pxr import Gf, Usd, UsdGeom
import torch
import omni.usd

# from isaaclab.command_generators import UniformPoseCommandGenerator

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    
def angle_to_target_observation(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    
    # cmd = env.command_manager.get_command(command_name)
    # test_pos_command_b = cmd[:, :2]
    # # print("angle to target observation: ", heading_command_b)  # Debugging output
    # angle = torch.atan2(test_pos_command_b[:, 1], test_pos_command_b[:, 0])
    # print("[TEST] angle: ", angle)
    cmd = env.command_manager.get_command(command_name)
    heading_command_b = cmd[:, 3]
    print("[INFO] heading_command_b: ", heading_command_b)
    return heading_command_b.unsqueeze(-1)

def distance_to_target_euclidean(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    
    cmd = env.command_manager.get_command(command_name)
    pos_command_b = cmd[:, :2]
    dist = torch.norm(pos_command_b, dim=1)
    print("[Obs] distance to target euclidean: ", dist)  # Debugging output
    
    return dist.unsqueeze(-1)

def angle_diff(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Calculate the angle difference between the rover's heading and the target."""
    
    target_heading_b = env.command_manager.get_command(command_name)[:, 4]
    # print("angle_diff: ", heading_angle_diff)  # Debugging output
    print("[TEST] target heading b: ", target_heading_b)
    
    return target_heading_b.unsqueeze(-1)
