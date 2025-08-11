from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pxr import Gf, Usd, UsdGeom
import math

import omni.usd

from isaaclab.assets import AssetBase, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

pre_dist = None

def distance_to_target_reward(
    env: ManagerBasedRLEnv,
    command_name: str
) -> torch.Tensor:

    cmd = env.command_manager.get_command(command_name)
    pos_local = cmd[:, :2]                        # (n_envs, 2)
    # print("[Reward] pos_local: ", pos_local)  # Debugging output
    distance  = torch.norm(pos_local, dim=-1)
    # print("[Reward] distance: ", distance)  # Debugging output

    k = 0.11
    reward = (1.0 / (1.0 + k * distance**2)) / env.max_episode_length
    # print(f"[Distance] distance: {distance}, [Reward] distance to target reward: {reward}")  # Debugging output

    return reward

def reached_target(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float,
    angle_threshold: float
) -> torch.Tensor:

    cmd = env.command_manager.get_command(command_name)  # (n_envs, N)

    target_position = cmd[:, :2]  # x_local, y_local
    # print("[Reward] target_position: ", target_position)  # Debugging output
    heading_cmd_w   = cmd[:, 3]   # heading_command_w

    distance = torch.norm(target_position, p=2, dim=-1)             # (n_envs,)
    # 수정 필요 
    angle_ok = torch.abs(heading_cmd_w) <= angle_threshold          # (n_envs,) within ~5.7°
    # if angle_ok == torch.zeros(heading_cmd_w):
    #     print(f"[INFO] angle_ok,, forklift 헤딩이랑 target point까지 목표 헤딩 차이: {heading_cmd_w}") 
    # print("[Reward] angle to goal, distance: ", distance)  
    # print("[Reward] angle to goal, angle: ", heading_cmd_w)  
    remaining      = env.max_episode_length - env.episode_length_buf
    reward_scale   = remaining / env.max_episode_length               # (n_envs,)

    reached = (distance <= distance_threshold) & angle_ok
    reward  = torch.where(reached, 2.0 * reward_scale, torch.zeros_like(reward_scale))
    # print("[Reward] reached target, distance: ", distance)  # Debugging output

    return reward

def angle_to_goal_reward(
        env: ManagerBasedRLEnv,
        command_name: str 
    ) -> torch.Tensor:

    cmd = env.command_manager.get_command(command_name)  # shape: (n_envs, N)

    pos_command_b = cmd[:, :2]     # (n_envs, 2)
    target_heading_b = cmd[:, 4]      # (n_envs,)

    distance     = torch.norm(pos_command_b, p=2, dim=-1)           # (n_envs,)
    angle_reward = (1 / (1 + distance)) * (1 / (1 + torch.abs(target_heading_b)))

    return angle_reward / env.max_episode_length

def heading_soft_contraint(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    # 후진하지 않으면 보상 없음(후진 유도)
    return torch.where(env.action_manager.action[:, 0] > 0.0, (1.0 / env.max_episode_length), 0.0)

def angle_to_target_penalty(
        env: ManagerBasedRLEnv,
        command_name: str,
        # angle_threshold: float
    ) -> torch.Tensor:

    cmd = env.command_manager.get_command(command_name)  # shape: (n_envs, N)

    heading_cmd_b = cmd[:, 3]                        # (n_envs, 2)

    abs_angle = torch.abs(heading_cmd_b)
    
    # too_far  = abs_angle > angle_threshold
    
    return torch.where(abs_angle > 2.0, abs_angle / env.max_episode_length, 0.0)

def far_from_target_reward(
        env: ManagerBasedRLEnv, 
        command_name: str, 
        distance_threshold: float
    ) -> torch.Tensor:

    cmd = env.command_manager.get_command(command_name)  # shape: (n_envs, N)

    target_pos_local = cmd[:, :2]                       # (n_envs, 2)

    distance = torch.norm(target_pos_local, p=2, dim=-1)  # (n_envs,)

    penalty = torch.where(
        distance > distance_threshold,
        torch.ones_like(distance),
        torch.zeros_like(distance)
    )

    return penalty