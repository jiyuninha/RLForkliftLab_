from typing import Sequence

import random
import math
import torch

from pxr import Sdf, Usd, UsdGeom, Gf
import omni.usd
import isaaclab.sim as sim_utils

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

# Import Pallet Scene 


# Visualization Marker Configuration
SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)
            ),
        ),
    }
)

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg


# Helper Functions (moved outside the class)
def quat_from_yaw():
    """Generate a random quaternion based on a yaw angle between -90° and 90°."""
    y = random.uniform(-math.pi/2, math.pi/2)
    w = math.cos(y / 2)
    z = math.sin(y / 2)
    return Gf.Quatd(w, Gf.Vec3d(0, 0, z))

def cal(distance, yaw):
    """
    Calculate the x and y offsets given a distance and a yaw angle (in degrees).
    """
    dx = math.sin(math.radians(yaw)) * distance
    dy = math.cos(math.radians(180 - yaw)) * distance
    return (dx, dy)

def decide(s0, s1, distance, yaw, pallet_offset=False):
    """
    s0: Forklift's (x, y) position.
    s1: Pallet's (x, y) position.
    
    Calculate two candidate goal points based on the pallet's yaw and return the one
    that is closer to the forklift.
    """
    if pallet_offset:
        distance += 0.4
    a = cal(distance, yaw)
    goal_point1 = (s1[0] - a[0], s1[1] - a[1])
    goal_point2 = (s1[0] + a[0], s1[1] + a[1])
    distance1 = math.dist(goal_point1, s0)
    distance2 = math.dist(goal_point2, s0)
    return goal_point1 if distance1 < distance2 else goal_point2

def quat2yaw(quat):
    """
    Convert a Gf.Quatd quaternion to a yaw angle (in degrees).
    """
    rotation = Gf.Rotation(quat)
    euler_angles = rotation.Decompose(
        Gf.Vec3d(1, 0, 0),
        Gf.Vec3d(0, 1, 0),
        Gf.Vec3d(0, 0, 1)
    )
    return euler_angles[2]  # Z-axis yaw

class TargetPalletCommand(CommandTerm):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.forklift: Articulation = env.scene[cfg.asset_name]
        self.stage = omni.usd.get_context().get_stage()
        self.pallet_prim_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/Pallet")

        self.target_distance = 2.0
        self._device = self.forklift.data.device

        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self._device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self._device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(1)), dim=1)

    def _update_metrics(self):
        self.metrics["error_pos"] = torch.norm(
            self.pos_command_w - self.forklift.data.root_link_pos_w[:, :3], dim=1)
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.forklift.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        # Step 1: 환경별 pallet 위치와 orientation 무작위 설정
        with Sdf.ChangeBlock():
            for i, path in enumerate(self.pallet_prim_paths):
                pallet_prim = self.stage.GetPrimAtPath(path)
                if not pallet_prim:
                    continue

                # 랜덤 위치
                translate_attr = pallet_prim.GetAttribute("xformOp:translate")
                if translate_attr is None:
                    translate_attr = pallet_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)

                rand_x = random.uniform(-3.0, 3.0)
                rand_y = random.uniform(-3.0, 3.0)
                translate_attr.Set(Gf.Vec3d(rand_x, rand_y, 0.0))

                # 랜덤 yaw
                orient_attr = pallet_prim.GetAttribute("xformOp:orient")
                if orient_attr is None:
                    orient_attr = pallet_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)

                orient_attr.Set(quat_from_yaw())

        # Step 2: pallet 기준으로 각 환경의 target point 설정
        for i, path in enumerate(self.pallet_prim_paths):
            pallet_prim = self.stage.GetPrimAtPath(path)
            if not pallet_prim:
                continue

            translate_attr = pallet_prim.GetAttribute("xformOp:translate")
            orient_attr = pallet_prim.GetAttribute("xformOp:orient")
            if translate_attr is None or orient_attr is None:
                continue

            translate = translate_attr.Get()
            orient = orient_attr.Get()
            yaw = quat2yaw(orient)

            s1 = (translate[0], translate[1])  # pallet 위치
            s0 = (0.0, 0.0)  # 기준점 (무시 가능)

            # target 후보 계산
            a = cal(self.target_distance, yaw)
            candidate1 = (s1[0] - a[0], s1[1] - a[1])
            candidate2 = (s1[0] + a[0], s1[1] + a[1])

            selected_xy = random.choice([candidate1, candidate2])
            selected_z = 0.0  # flat ground 가정

            selected_target = torch.tensor([*selected_xy, selected_z], device=self._device)

            if i in env_ids:
                self.pos_command_w[i] = selected_target
                self.heading_command_w[i] = 0.0

    def _update_command(self):
        target_vec = self.pos_command_w - self.forklift.data.root_link_pos_w[:, :3]
        self.pos_command_b = quat_rotate_inverse(yaw_quat(self.forklift.data.root_link_quat_w), target_vec)
        self.heading_command_b = wrap_to_pi(self.heading_command_w - self.forklift.data.heading_w)



if __name__ == "__main__":
    importer = TargetPalletCommand()
    new_targets = importer.sample_new_target()
    print("Target points for each environment:", new_targets)
