from typing import Sequence
import torch
import math
import random
from pxr import UsdGeom, Gf, Usd
import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

# --- Quaternion and angle utility functions ---

def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return ((x + math.pi) % (2 * math.pi)) - math.pi


# def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
#     half  = yaw * 0.5
#     cos_h = torch.cos(half)
#     sin_h = torch.sin(half)
#     zeros = torch.zeros_like(cos_h)
#     return torch.stack([cos_h, zeros, zeros, sin_h], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    orig_shape = q.shape
    q_flat    = q.reshape(-1, 4)
    w, x, y, z = q_flat.unbind(1)
    qc_flat   = torch.stack([w, -x, -y, -z], dim=1)
    return qc_flat.view(orig_shape)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    s1       = q1.shape
    q1_flat  = q1.reshape(-1, 4)
    q2_flat  = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1_flat.unbind(1)
    w2, x2, y2, z2 = q2_flat.unbind(1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    out_flat = torch.stack([w, x, y, z], dim=1)
    return out_flat.view(s1)


# def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#     if q.dim() == 1:
#         q = q.unsqueeze(0).expand(v.shape[0], -1)

#     zeros   = torch.zeros(v.shape[:-1] + (1,), device=v.device, dtype=v.dtype)
#     v_quat  = torch.cat([zeros, v], dim=-1)

#     qc   = quat_conjugate(q)
#     tmp  = quat_mul(qc, v_quat)
#     res  = quat_mul(tmp, q)

#     return res[..., 1:4]


def quat2yaw(quat: Gf.Quatd) -> float:
    w        = quat.GetReal()
    x, y, z  = quat.GetImaginary()
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)
    )

class TargetPalletCommand(CommandTerm):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.forklift: Articulation = env.scene[cfg.asset_name]
        self.stage = omni.usd.get_context().get_stage()
        self._xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        self.target_prim_paths = sim_utils.find_matching_prim_paths(
            "/World/envs/env_.*/Target"
        )

        self._device = self.forklift.data.device

        # World-frame commands
        self.pos_command_w     = torch.zeros(self.num_envs, 3, device=self._device)
        self.heading_command_w = torch.zeros(self.num_envs,   device=self._device)

        # Body-frame commands
        self.pos_command_b     = torch.zeros_like(self.pos_command_w)
        self.test_pos_command_b     = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        # Target point heading
        self.target_heading_w  = torch.zeros(self.num_envs,   device=self._device)
        self.target_heading_b  = torch.zeros(self.num_envs, device=self._device)

        # Forklift heading (world-frame, corrected +π)
        self.forklift_heading_w = torch.zeros(self.num_envs, device=self._device)

        # For Debugging
        self.pallet_prim_paths = sim_utils.find_matching_prim_paths(
            "/World/envs/env_.*/Pallet"
        )
        self.pallet_pos_w = torch.zeros(self.num_envs, 3, device=self._device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([
            self.test_pos_command_b,                 # (n,3)
            self.heading_command_b.unsqueeze(1), # (n,1)
            # self.heading_command_w.unsqueeze(1),
            self.target_heading_b.unsqueeze(1),  # (n,1)
            self.forklift_heading_w.unsqueeze(1), # (n,1) 추가
        ], dim=1)

    def _update_metrics(self):
        self.metrics["error_pos"] = torch.norm(
            self.pos_command_w - self.forklift.data.root_link_pos_w[:, :3], dim=1)
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.forklift_heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        """
        reset 후에 호출되는 함수로 변경된 Target point에 대한 값을 가져옴
        """
        self._xform_cache.Clear()

        # print("[Command] TargetPalletCommand: _resample_command")
        for path in self.target_prim_paths:
            parts = path.split("/")
            if len(parts) < 4 or not parts[3].startswith("env_"):
                continue

            env_idx = int(parts[3].split("_", 1)[1])
            if env_idx not in env_ids:
                continue

            prim = self.stage.GetPrimAtPath(path)
            if not prim:
                continue

            world_mat = self._xform_cache.GetLocalToWorldTransform(prim)
            world_pos = world_mat.ExtractTranslation()  # Gf.Vec3d

            self.pos_command_w[env_idx] = torch.tensor(
                [world_pos[0], world_pos[1], world_pos[2]],
                device=self._device,
            )

            orient = prim.GetAttribute("xformOp:orient").Get()  # Gf.Quatd
            self.target_heading_w[env_idx] = quat2yaw(orient)

        # For Debugging
        for path in self.pallet_prim_paths:
            parts = path.split("/")
            if len(parts) < 4 or not parts[3].startswith("env_"):
                continue
            env_idx = int(parts[3].split("_",1)[1])
            if env_idx not in env_ids:
                continue

            prim = self.stage.GetPrimAtPath(path)
            if not prim:
                continue
            world_mat = self._xform_cache.GetLocalToWorldTransform(prim)
            wp = world_mat.ExtractTranslation()  # Gf.Vec3d
            # tensor 로 변환해서 저장
            self.pallet_pos_w[env_idx] = torch.tensor(
                [wp[0], wp[1], wp[2]],
                device=self._device
            )

    def _update_command(self):
        """
        포크리프트 헤딩에 180° 옵셋 적용
        """
        self.forklift_heading_w = wrap_to_pi(
            self.forklift.data.heading_w + math.pi
        )

        """
        매 스텝마다 변경되는 forklift의 위치를 반영하여 heading_command_w와 heading_command_b를 구함
        """
        
        forklift_xy = self.forklift.data.root_link_pos_w[:, :2]   # (N,2)
        target_xy   = self.pos_command_w[:, :2]                  # (N,2)

        target2vec  = target_xy - forklift_xy                    # (N,2)
        target_vec = self.pos_command_w - self.forklift.data.root_link_pos_w[:, :3]
        self.test_pos_command_b[:] = quat_rotate_inverse(
        yaw_quat(self.forklift.data.root_link_quat_w), target_vec)
        self.heading_command_w = torch.atan2(target2vec[:,1], target2vec[:,0])
        """
        pos_command_b = (x_local, y_local, z_local) -> 포크리프트 정면 방향(+x) 기준으로 목표가 얼마나 떨어져 있는지, 포크리프트 옆면 방향(+y) 기준으로 목표가 얼마나 떨어져 있는지
        """
        zeros_z = torch.zeros(target2vec.shape[0], 1,
                              device=self._device, dtype=target2vec.dtype)  # (N,1)
        delta3d = torch.cat([target2vec, zeros_z], dim=1)                    # (N,3)
        self.pos_command_b = quat_rotate_inverse(
            self.forklift.data.root_link_quat_w,
            delta3d) 
        # print("[INFO] pos_command_b: ", self.pos_command_b) # forklift의 body frame 기준으로 계산된 목표 위치까지의 3D 이동 벡터
        self.heading_command_b = wrap_to_pi(
            self.heading_command_w - self.forklift_heading_w)
        # print("heading_command_b: ", self.heading_command_b) # 포크리프트의 body frame 기준으로 목표 heading이 얼마나 떨어져 있는지

        """
        target point의 heading을 world frame에서 body frame(forklift) 으로 변환 
        """
        self.target_heading_b = wrap_to_pi(self.target_heading_w - self.forklift_heading_w)
        # print("[TEST] target_heading_b: ", self.target_heading_b)