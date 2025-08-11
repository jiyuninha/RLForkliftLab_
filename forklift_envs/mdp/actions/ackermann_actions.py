from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv  # noqa: F811

    from . import actions_cfg


class AckermannAction(ActionTerm):

    cfg: actions_cfg.AckermannActionCfg

    _asset: Articulation

    _wheelbase_length: float

    _middle_wheel_distance: float

    _rear_and_front_wheel_distance: float

    _wheel_radius: float

    _min_steering_radius: float

    _steering_joint_names: list[str]

    _drive_joint_names: list[str]

    _scale: torch.Tensor

    _offset: torch.Tensor

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(self.cfg.steering_joint_names)

        # remap joints to the order specified in the config.
        steering_order = cfg.steering_order
        drive_order = cfg.drive_order
        sorted_steering_joint_names = sorted(self._steering_joint_names, key=lambda x: steering_order.index(x))
        sorted_drive_joint_names = sorted(self._drive_joint_names, key=lambda x: drive_order.index(x))
        original_steering_id_positions = {name: i for i, name in enumerate(self._steering_joint_names)}
        original_drive_id_positions = {name: i for i, name in enumerate(self._drive_joint_names)}
        self._sorted_steering_ids = [self._steering_joint_ids[original_steering_id_positions[name]]
                                     for name in sorted_steering_joint_names]
        self._sorted_drive_ids = [self._drive_joint_ids[original_drive_id_positions[name]]
                                  for name in sorted_drive_joint_names]

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, len(self._steering_joint_ids), device=self.device)

        # Save the scale and offset for the actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        # print("[AckermannAction] process_actions: ", actions)
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):

        self._joint_pos, self._joint_vel = ackermann(
            self._processed_actions[:, 0], self._processed_actions[:, 1], self.cfg, self.device)

        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)
        # print("[AckermannAction] joint_vel: ", self._joint_vel)
        self._asset.set_joint_position_target(self._joint_pos, joint_ids=self._steering_joint_ids)
        # print("[AckermannAction] joint_pos: ", self._joint_pos)


class AckermannActionNonVec():
    def __init__(self,
                 cfg: actions_cfg.AckermannActionCfg,
                 robot: Articulation,
                 num_envs: int,
                 device: torch.device):
        """ Initialize the AckermannActionNonVec

        Args:
            cfg (actions_cfg.AckermannActionCfg): configuration for the ackermann action
            robot (Articulation): robot asset
            num_envs (int): number of environments
            device (torch.device): device to run the operation on
        """
        # Initialize Parameters
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._asset = robot

        # Find the joint ids and names for the drive and steering joints
        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(self.cfg.steering_joint_names)

        # Remap joints to the order specified in the config.
        steering_order = cfg.steering_order
        drive_order = cfg.drive_order
        sorted_steering_joint_names = sorted(self._steering_joint_names, key=lambda x: steering_order.index(x))
        sorted_drive_joint_names = sorted(self._drive_joint_names, key=lambda x: drive_order.index(x[:2]))
        original_steering_id_positions = {name: i for i, name in enumerate(self._steering_joint_names)}
        original_drive_id_positions = {name: i for i, name in enumerate(self._drive_joint_names)}
        self._sorted_steering_ids = [self._steering_joint_ids[original_steering_id_positions[name]]
                                     for name in sorted_steering_joint_names]
        self._sorted_drive_ids = [self._drive_joint_ids[original_drive_id_positions[name]]
                                  for name in sorted_drive_joint_names]

        carb.log_info(
            f" {self._drive_joint_ids} [{self._drive_joint_names}]"
            f" {self._steering_joint_ids} [{self._steering_joint_names}]"
        )

        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, len(self._steering_joint_ids), device=self.device)

        # Save the scale and offset for the actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # Store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    # def apply_actions(self):
    #     # Apply the actions to the rover
    #     self._joint_pos, self._joint_vel = ackermann(
    #         -self._processed_actions[:, 0], self._processed_actions[:, 1], self.cfg, self.device)

    #     self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._sorted_drive_ids)
    #     self._asset.set_joint_position_target(self._joint_pos, joint_ids=self._sorted_steering_ids)
    def apply_actions(self):
        # processed_actions: (n_envs, 2) → [:,0]=v, [:,1]=δ
        v, delta = self._processed_actions[:,0], self._processed_actions[:,1]
        # 1) 스티어링 각도: (n_envs,1)
        steer_pos = delta.unsqueeze(1)
        # 2) 드라이브 휠 각속도: (n_envs,1)
        #    wheel_radius는 cfg에 저장해 두셨을 거예요.
        wheel_vel = (v / self.cfg.wheel_radius).unsqueeze(1)
        # 3) 플랫폼에 바로 적용
        #    steering_joint_ids: 한 개 혹은 두 개가 같은 축에 묶여 있을 겁니다.
        self._asset.set_joint_position_target(steer_pos,
                                              joint_ids=self._steering_joint_ids)
        #    drive_joint_ids: 하나의 휠 모터 ID
        self._asset.set_joint_velocity_target(wheel_vel,
                                              joint_ids=self._drive_joint_ids)


def ackermann(lin_vel, ang_vel, cfg, device):
    """ Ackermann steering model for the rover
    Args:
        lin_vel (torch.Tensor): linear velocity of the rover
        ang_vel (torch.Tensor): angular velocity of the rover
        cfg (actions_cfg.AckermannActionCfg): configuration for the ackermann action
        device (torch.device): device to run the operation on

    Returns:
        torch.Tensor: steering angles for the rover
        torch.Tensor: wheel velocities for the rover
    """
    # forklift의 물리적 파라미터
    wheel_radius = 0.2          # 바퀴 반지름 (m)
    L = 0.6                 # 전륜과 후륜 사이의 거리 (m)
    eps = 1e-6

    # 선형 속도가 0이 아닌 경우 turning radius 계산, 0인 경우 무한대로 처리
    turning_radius = torch.where(
        torch.abs(ang_vel) > eps,
        lin_vel / (ang_vel + eps),
        torch.full_like(lin_vel, float('inf'))
    )

    # forklift의 경우, steer angle은 보통 앞바퀴를 제어하므로:
    # turning_radius가 inf (즉, 직진)일 경우 steer angle은 0,
    # 그 외에는 아크탄젠트로 휨 각도를 계산합니다.
    steering_angle = torch.where(
        torch.isinf(turning_radius),
        torch.zeros_like(turning_radius),
        torch.atan(L / (turning_radius + eps))
    )
    
    # 후륜의 구동 속도: 선형 속도를 바퀴 반지름으로 나누어 rad/s로 변환
    wheel_velocity = lin_vel / (wheel_radius + eps)
    
    # 각 값은 배치당 단일 값이도록 unsqueeze (예: shape (batch,) -> (batch, 1))
    steering_angle = steering_angle.unsqueeze(1)
    driving_wheel = wheel_velocity.unsqueeze(1)

    # print("steering angle: ", steering_angle)
    # print("driving wheel: ", driving_wheel.shape)

    # test_steer = torch.full((1,1), 0.0, device='cuda:0')
    # test_drive = torch.full((1,1), 0.5, device='cuda:0')

    # return test_steer, test_drive
    return  steering_angle, driving_wheel