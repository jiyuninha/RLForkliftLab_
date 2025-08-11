from typing import TYPE_CHECKING  # noqa: F401

import torch
import random
import math
from typing import Dict, Tuple
import numpy as np

from pxr import Sdf, UsdGeom, Gf, Usd
import omni.usd

from isaaclab.assets import RigidObject, AssetBase, AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_apply, wrap_to_pi

from forklift_envs.envs.local_navigation.utils.pallets.commands_cfg import TargetPalletCommandCfg  # noqa: F401
from forklift_envs.envs.local_navigation.utils.pallets.pallets_importer import TargetPalletCommand  # noqa: F40


def reset_root_state_forklift(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, z_offset: float = 0.0
):
    # print("[Debug] Event: reset_root_state_forklift")
    asset: RigidObject = env.scene[asset_cfg.name]

    n_env = len(env_ids)
    device = env.device

    origins = env.scene.terrain.env_origins[env_ids]  # shape: [n_env, 3]

    positions = origins.clone()
    # positions[:, 2] += z_offset

    orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n_env, 1)

    asset.write_root_link_pose_to_sim(
        torch.cat([positions, orientations], dim=-1),
        env_ids=env_ids
    )

def _dist2_xy(v: Gf.Vec3d) -> float:
    return v[0] * v[0] + v[1] * v[1]

def reset_root_state_pallet_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pallet_cfg: AssetBaseCfg,
    target_cfg: AssetBaseCfg,
    z_offset: float = 0.5
):
    # print("[Reset]")
    if env_ids is None:
        total_envs = env.scene.terrain.env_origins.shape[0]
        env_ids = torch.arange(total_envs, dtype=torch.long, device=env.device)

    stage = omni.usd.get_context().get_stage()
    pallet_asset: AssetBase = env.scene[pallet_cfg.name]
    target_asset: AssetBase = env.scene[target_cfg.name]

    pallet_prim_paths = pallet_asset.prim_paths
    target_prim_paths = target_asset.prim_paths

    ori_map: Dict[int, Gf.Quatd] = {}
    pallet_pos: Dict[int, Gf.Vec3d] = {}

    reset_envs = set(env_ids.tolist())

    fixed_pallet_pos = Gf.Vec3d(-6.0, 0.0, 0.0)

    with Sdf.ChangeBlock():
        for pallet_prim_path in pallet_prim_paths:
            parts = pallet_prim_path.split("/")
            if len(parts) < 5 or not parts[3].startswith("env_"):
                continue
            try:
                env_index = int(parts[3].split("_", 1)[1])
            except ValueError:
                continue

            if env_index not in reset_envs:
                continue

            pallet_prim = stage.GetPrimAtPath(pallet_prim_path)

            if not pallet_prim:
                continue

            pallet_xform = UsdGeom.Xform(pallet_prim)
            pallet_ops = pallet_xform.GetOrderedXformOps()
            translate_op = next((o for o in pallet_ops if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            orient_op = next((o for o in pallet_ops if o.GetOpType() == UsdGeom.XformOp.TypeOrient), None)

            if translate_op:
                translate_op.Set(fixed_pallet_pos) 
            else:
                pallet_xform.AddTranslateOp().Set(fixed_pallet_pos)

            pallet_pos[env_index] = fixed_pallet_pos
            angle_deg = torch.empty((), device="cuda:0").uniform_(45.0, 135.0)
            yaw = angle_deg * math.pi / 180.0
            qw = math.cos(yaw / 2.0)
            qz = math.sin(yaw / 2.0)
            pallet_quat = Gf.Quatd(qw, Gf.Vec3d(0.0, 0.0, qz))

            ori_map[env_index] = pallet_quat

            if orient_op:
                orient_op.Set(pallet_quat)
            else:
                pallet_xform.AddOrientOp().Set(pallet_quat)
        
        
        forward_local = torch.tensor([0.0, 1.0, 0.0], device=env.device, dtype=torch.float32)  # +Y 전방이면 [0,1,0]
        front_dist = 2.0
        z_off = float(z_offset)
        
        # Update target prim
        for prim_path in target_prim_paths:
            parts = prim_path.split("/")
            if len(parts) < 5 or not parts[3].startswith("env_"):
                continue
            env_idx = int(parts[3].split("_",1)[1])
            if env_idx not in reset_envs:
                continue

            # 1) 팔레트의 "env별" 월드 위치/회전 가져오기
            p_pos = pallet_pos[env_idx]          # Gf.Vec3d
            q_gf  = ori_map[env_idx]             # Gf.Quatd (w, xyz)

            # 2) Gf.Quatd -> torch(w,x,y,z) 변환
            qw = float(q_gf.GetReal())
            qx, qy, qz = [float(v) for v in q_gf.GetImaginary()]
            q_torch = torch.tensor([qw, qx, qy, qz], device=env.device, dtype=torch.float32)

            # 3) 로컬 전방벡터(+X 가정)를 쿼터니언으로 회전 → 월드 전방
            dir_world = quat_apply(q_torch, forward_local)  # (3,)
            dx, dy = float(dir_world[0]), float(dir_world[1])

            # 4) 팔레트 앞/뒤 후보 (팔레트 "자기" 위치 기준)
            pos_plus  = Gf.Vec3d(p_pos[0] + dx * front_dist, p_pos[1] + dy * front_dist, p_pos[2] + z_off)
            pos_minus = Gf.Vec3d(p_pos[0] - dx * front_dist, p_pos[1] - dy * front_dist, p_pos[2] + z_off)

            # 5) 지게차와의 거리로 앞/뒤 선택 (원점이 아니라 지게차 좌표!)
            fl_xy_tensor = env.scene["forklift"].data.root_link_pos_w[env_idx, :2]
            fx, fy = fl_xy_tensor[0].item(), fl_xy_tensor[1].item()

            d2_plus = pos_plus[0] ** 2 + pos_plus[1] ** 2
            d2_minus = pos_minus[0] ** 2 + pos_minus[1] ** 2

            target_pos = pos_plus if d2_plus <= d2_minus else pos_minus

            # 6) 타깃이 팔레트를 바라보도록 yaw 계산 → 쿼터니언
            face_yaw = math.atan2(p_pos[1] - target_pos[1], p_pos[0] - target_pos[0])
            qw_t, qz_t = math.cos(face_yaw * 0.5), math.sin(face_yaw * 0.5)
            tgt_quat = Gf.Quatd(qw_t, Gf.Vec3d(0.0, 0.0, qz_t))

            # 7) USD에 적용
            prim  = stage.GetPrimAtPath(prim_path)
            xform = UsdGeom.Xform(prim)
            ops   = xform.GetOrderedXformOps()
            trans = next((o for o in ops if o.GetOpType()==UsdGeom.XformOp.TypeTranslate), None)
            orient= next((o for o in ops if o.GetOpType()==UsdGeom.XformOp.TypeOrient), None)
            (trans or xform.AddTranslateOp()).Set(target_pos)
            (orient or xform.AddOrientOp()).Set(tgt_quat)