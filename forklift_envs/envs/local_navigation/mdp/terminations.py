from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import Gf, Usd, UsdGeom

import omni.usd

# Importing necessary modules from the isaaclab package
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation, RigidObject, AssetBase
from isaaclab.utils.math import wrap_to_pi


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float,
    angle_threshold: float
) -> torch.Tensor:    

    cmd = env.command_manager.get_command(command_name)  # (n_envs, N)

    target_pos_local = cmd[:, :2]    # (n_envs, 2)
    heading_cmd_w    = cmd[:, 3]     # (n_envs,)

    distance = torch.norm(target_pos_local, p=2, dim=-1)      # (n_envs,)
    angle_ok = torch.abs(heading_cmd_w) <= angle_threshold   # (n_envs,)

    success = (distance <= distance_threshold) & angle_ok    # (n_envs,)

    return success



def far_from_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float
) -> torch.Tensor:

    cmd = env.command_manager.get_command(command_name)  # shape: (n_envs, N)
    target_pos_local = cmd[:, :2]                       # (n_envs, 2)
    distance = torch.norm(target_pos_local, p=2, dim=-1) # (n_envs,)
    is_far = distance > distance_threshold

    return is_far


# 아직 구현 안 함!
# def collision_with_obstacles(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
#     """
#     Checks for collision with obstacles.
#     """

#     # print(f"[Termination] collision with obstacles, sensor name is {sensor_cfg.name}")
#     # Accessing the contact sensor and its data
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

#     # Reshape as follows (num_envs, num_bodies, 3)
#     force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)
#     # print(f"[INFO] force matrix: {force_matrix}")
#     # Calculating the force and returning true if it is above the threshold
#     normalized_forces = torch.norm(force_matrix, dim=1)
#     forces_active = torch.sum(normalized_forces, dim=-1) > 1

#     return torch.where(forces_active, True, False)
