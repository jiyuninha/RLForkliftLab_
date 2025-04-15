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


# class TargetBasedPositionCommand(CommandTerm):
#     """Command generator that generates position commands based on the terrain.

#     The position commands are sampled from the terrain mesh and the heading commands are either set
#     to point towards the target or are sampled uniformly.
#     """

#     """Configuration for the command generator."""

#     def __init__(self, cfg, env: ManagerBasedEnv):
#         """Initialize the command generator class.

#         Args:
#             cfg: The configuration parameters for the command generator.
#             env: The environment object.
#         """
#         # initialize the base class
#         super().__init__(cfg, env)

#         # obtain the robot and terrain assets
#         # -- robot
#         self.forklift: Articulation = env.scene[cfg.asset_name]
#         # -- terrain
#         self.terrain: PalletsSceneImporter = env.scene.terrain 

#         # crete buffers to store the command
#         # -- commands: (x, y, z, heading)
#         self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
#         self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
#         self.pos_command_b = torch.zeros_like(self.pos_command_w)
#         self.heading_command_b = torch.zeros_like(self.heading_command_w)
#         # -- metrics
#         self.metrics["error_pos"] = torch.zeros(
#             self.num_envs, device=self.device)
#         self.metrics["error_heading"] = torch.zeros(
#             self.num_envs, device=self.device)

#     def __str__(self) -> str:
#         msg = "TargetBasedPositionCommand:\n"
#         msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
#         msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
#         msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
#         return msg

#     """
#     Properties
#     """

#     @property
#     def command(self) -> torch.Tensor:
#         """The desired base position in base frame. Shape is (num_envs, 3)."""
#         return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(1)), dim=1)

#     """
#     Implementation specific functions.
#     """

#     def _resample_command(self, env_ids: Sequence[int]):
#         # sample new position targets from the terrain
#         self.pos_command_w[env_ids] = self.terrain.sample_new_target(env_ids)
#         # offset the position command by the current root position
#         self.pos_command_w[env_ids,
#                            2] += self.robot.data.default_root_state[env_ids, 2]
#         # random heading command
#         r = torch.empty(len(env_ids), device=self.device)
#         self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

#     def _update_command(self):
#         """Re-target the position command to the current root position and heading."""
#         target_vec = self.pos_command_w - \
#             self.robot.data.root_link_pos_w[:, :3]
#         self.pos_command_b[:] = quat_rotate_inverse(
#             yaw_quat(self.robot.data.root_link_quat_w), target_vec)
#         self.heading_command_b[:] = wrap_to_pi(
#             self.heading_command_w - self.robot.data.heading_w)

#     def _update_metrics(self):
#         # logs data
#         self.metrics["error_pos"] = torch.norm(
#             self.pos_command_w - self.robot.data.root_link_pos_w[:, :3], dim=1)
#         self.metrics["error_heading"] = torch.abs(wrap_to_pi(
#             self.heading_command_w - self.robot.data.heading_w))

#     def _set_debug_vis_impl(self, debug_vis: bool):

#         if debug_vis:
#             if not hasattr(self, "arrow_goal_visualizer"):
#                 arrow_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
#                 arrow_cfg.prim_path = "/Visuals/Command/heading_goal"
#                 arrow_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)
#                 self.arrow_goal_visualizer = VisualizationMarkers(arrow_cfg)
#             if not hasattr(self, "sphere_goal_visualizer"):
#                 sphere_cfg = SPHERE_MARKER_CFG.copy()
#                 sphere_cfg.prim_path = "/Visuals/Command/position_goal"
#                 sphere_cfg.markers["sphere"].radius = 0.2
#                 self.sphere_goal_visualizer = VisualizationMarkers(sphere_cfg)

#             # set their visibility to true
#             self.arrow_goal_visualizer.set_visibility(True)
#             self.sphere_goal_visualizer.set_visibility(True)
#         else:
#             if hasattr(self, "arrow_goal_visualizer"):
#                 self.arrow_goal_visualizer.set_visibility(False)
#             if hasattr(self, "sphere_goal_visualizer"):
#                 self.sphere_goal_visualizer.set_visibility(False)

#     def _debug_vis_callback(self, event):
#         # update the sphere marker
#         self.sphere_goal_visualizer.visualize(self.pos_command_w)

#         # update the arrow marker
#         zero_vec = torch.zeros_like(self.heading_command_w)
#         quaternion = quat_from_euler_xyz(
#             zero_vec, zero_vec, self.heading_command_w)
#         position_arrow_w = self.pos_command_w + \
#             torch.tensor([0.0, 0.0, 0.25], device=self.device)
#         self.arrow_goal_visualizer.visualize(position_arrow_w, quaternion)

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

# # Main Class: PalletsSceneImporter
# class PalletsSceneImporter(TerrainImporter):
#     def __init__(self, cfg: TerrainImporterCfg):
#         super().__init__(cfg)
#         """
#         cfg:
#             - num_envs: Number of environments.
#             - pallet_prim_expr: Regex pattern for pallet prim paths.
#             - forklift_prim_expr: Regex pattern for forklift prim paths.
#             - target_distance: Baseline distance for generating the goal.
#         """
#         self.cfg = cfg
#         # self._terrainManager = TerrainManager(
#         #     num_envs=self._cfg.num_envs, device=self.device)
#         self.num_envs = cfg.num_envs
#         self.target_distance = getattr(cfg, "target_distance", 1.0)
        
#         # Obtain USD Stage and prim paths based on provided patterns.
#         self.stage = omni.usd.get_context().get_stage()
#         self.pallet_prim_paths = sim_utils.find_matching_prim_paths(cfg.pallet_prim_expr)
#         self.forklift_prim_paths = sim_utils.find_matching_prim_paths(cfg.forklift_prim_expr)
        
#         # List to store target points for each environment.
#         self.target_points = [None] * self.num_envs
        
#         # Initialize sphere marker visualizer for target point visualization.
#         self.sphere_goal_visualizers = []
#         for i in range(cfg.num_envs):
#             sphere_cfg = SPHERE_MARKER_CFG.copy()
#             sphere_cfg.markers["sphere"].radius = 0.05
#             sphere_cfg.prim_path = f"/World/envs/env_{i}/position_goal"
#             marker = VisualizationMarkers(sphere_cfg)
#             #marker.set_visibility(True)
#             self.sphere_goal_visualizers.append(marker)

#     def randomization_pallet(self):
#         """
#         Update the orientation of all pallet prims with a random yaw-based quaternion.
#         """
#         with Sdf.ChangeBlock():
#             for prim_path in self.pallet_prim_paths:
#                 pallet_prim = self.stage.GetPrimAtPath(prim_path)
#                 if not pallet_prim:
#                     continue
#                 orient_attr = pallet_prim.GetAttribute("xformOp:orient")
#                 if orient_attr is None:
#                     pallet_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)
#                     orient_attr = pallet_prim.GetAttribute("xformOp:orient")
#                 orient_attr.Set(quat_from_yaw())

#     def sample_new_target(self, env_ids: Sequence[int]) -> torch.Tensor:
#         """
#         1. Randomize the pallet orientations.
#         2. For each environment, retrieve the positions of the pallet and forklift.
#         3. Extract the pallet's yaw from its quaternion.
#         4. Compute two candidate target points and select the one closer to the forklift.
#         5. Visualize the selected target points using sphere markers.
#         """
#         self.randomization_pallet()
#         target_points = []
        
#         # Assume that pallet and forklift prim paths are ordered correspondingly for each environment.
#         for pallet_path, forklift_path in zip(self.pallet_prim_paths, self.forklift_prim_paths):
#             pallet_prim = self.stage.GetPrimAtPath(pallet_path)
#             forklift_prim = self.stage.GetPrimAtPath(forklift_path)
#             if not pallet_prim or not forklift_prim:
#                 target_points.append(None)
#                 continue

#             pallet_translate_attr = pallet_prim.GetAttribute("xformOp:translate")
#             forklift_translate_attr = forklift_prim.GetAttribute("xformOp:translate")
#             if pallet_translate_attr is None or forklift_translate_attr is None:
#                 target_points.append(None)
#                 continue

#             pallet_translate = pallet_translate_attr.Get()  # (x, y, z)
#             forklift_translate = forklift_translate_attr.Get()  # (x, y, z)

#             pallet_orient_attr = pallet_prim.GetAttribute("xformOp:orient")
#             if pallet_orient_attr is None:
#                 target_points.append(None)
#                 continue
#             pallet_orient = pallet_orient_attr.Get()
#             yaw = quat2yaw(pallet_orient)

#             s0 = forklift_translate[:2]  # (x, y)
#             s1 = pallet_translate[:2]      # (x, y)
#             goal_xy = decide(s0, s1, self.target_distance, yaw, pallet_offset=False)
#             # Use the forklift's z value (adjust if needed)
#             goal_point = (goal_xy[0], goal_xy[1], forklift_translate[2])
#             target_points.append(goal_point)
        
#         self.target_points = target_points

#         # Convert valid target points to a torch.Tensor and visualize them.
#         for i, pt in enumerate(self.target_points):
#             if pt is not None:
#                 pos_tensor = torch.tensor(pt, dtype=torch.float32)
#                 self.sphere_goal_visualizers[i].visualize(pos_tensor.unsqueeze(0))
        
#         return target_points

# # Example Configuration and Usage
# class Config:
#     num_envs = 5
#     pallet_prim_expr = "/World/Pallets/.*"       # Example: pallet prim path pattern
#     forklift_prim_expr = "/World/Forklifts/.*"    # Example: forklift prim path pattern
#     target_distance = 5.0

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
