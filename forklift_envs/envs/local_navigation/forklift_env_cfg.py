from __future__ import annotations

import math
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: F401
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: F401
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg
from isaaclab.sim import SimulationCfg as SimCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: F401
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: F401

##
# Scene Description
##
import forklift_envs
import forklift_envs.envs.local_navigation.mdp as mdp
from forklift_envs.envs.local_navigation.utils.pallets.pallet_scene import PalletSceneCfg  # noqa: F401
from forklift_envs.envs.local_navigation.utils.pallets.commands_cfg import TargetPalletCommandCfg  # noqa: F401
from forklift_envs.envs.local_navigation.utils.pallets.pallets_importer import TargetPalletCommand  # noqa: F401
from forklift_envs.envs.local_navigation.utils.articulation.articulation import ForkliftArticulation


from forklift_envs.assets.robots.forklift import FORKLIFT_CFG

@configclass
class ForkliftSceneCfg(PalletSceneCfg):
    """
    Rover Scene Configuration

    Note:
        Terrains can be changed by changing the parent class e.g.
        RoverSceneCfg(MarsTerrainSceneCfg) -> RoverSceneCfg(DebugTerrainSceneCfg)

    """
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )   

    forklift = FORKLIFT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Forklift",
        class_type=ForkliftArticulation
    )

    contact_sensor_lift = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Forklift/lift",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Pallet"])
    contact_sensor_body = ContactSensorCfg(
    	prim_path="{ENV_REGEX_NS}/Forklift/body",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Pallet"])
    
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )




@configclass
class ActionsCfg:
    """Action"""

    # We define the action space for the forklift
    actions: ActionTerm = MISSING


@configclass
class ObservationCfg:
    print("Observation")

    @configclass
    class PolicyCfg(ObsGroup):
        """ Policy """    
        actions = ObsTerm(func=mdp.last_action)
        distance = ObsTerm(func=mdp.distance_to_target_euclidean, 
                           params={
                                    "command_name": "target_pose",
                                    }, scale=0.11)
        heading = ObsTerm(
            func=mdp.angle_to_target_observation,
            params={
                "command_name": "target_pose",
            },
            scale=1 / math.pi
        )
        angle_diff = ObsTerm(
            func=mdp.angle_diff,
            params={
                "command_name": "target_pose",
                },
            scale=1 / math.pi
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    print("Reward")
    distance_to_target = RewTerm(
        func=mdp.distance_to_target_reward,
        weight=5.0,
        params={
            "command_name": "target_pose",
        },
    )
    reached_target = RewTerm(
        func=mdp.reached_target,
        weight=5.0,
        params={
            "command_name": "target_pose", 
            "distance_threshold": 0.1,  # 0.8m
            "angle_threshold": 0.1,  # ~5.7 degrees
        },
    )
    angle_to_target = RewTerm(
        func=mdp.angle_to_target_penalty,
        weight=-1.5,
        params={
            "command_name": "target_pose",
        },
    )
    heading_soft_contraint = RewTerm(
        func=mdp.heading_soft_contraint,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(name="forklift")
        },
    )
    far_from_target = RewTerm(
        func=mdp.far_from_target_reward,
        weight=-2.0,
        params={ 
            "command_name": "target_pose",
            "distance_threshold": 10.0,  # 5m
        },
    )
    angle_diff = RewTerm(
        func=mdp.angle_to_goal_reward,
        weight=5.0,
        params={
            "command_name": "target_pose", 
        },
    )


@configclass
class TerminationsCfg:
    print("Termination")

    time_limit = DoneTerm(func=mdp.time_out, time_out=True)
    is_success = DoneTerm(
        func=mdp.is_success,
        params={
            "command_name": "target_pose", 
            "distance_threshold": 0.1, 
            "angle_threshold": 0.1},
    )
    far_from_target = DoneTerm(
        func=mdp.far_from_target,
        params={ 
            "command_name": "target_pose",
            "distance_threshold": 10.0
        },
    )
    # collision = DoneTerm(
    #     func=mdp.collision_with_obstacles,
    #     params={"sensor_cfg": SceneEntityCfg(
    #         "contact_sensor_lift"), "threshold": 1.0},
    # )
    # collision = DoneTerm(
    #     func=mdp.collision_with_obstacles,
    #     params={"sensor_cfg": SceneEntityCfg(
    #         "contact_sensor_body"), "threshold": 1.0},
    # )


# "mdp.illegal_contact
@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    target_pose = TargetPalletCommandCfg(
        class_type=TargetPalletCommand,  # TerrainBasedPositionCommandCustom,
        asset_name="forklift",
        rel_standing_envs=0.0,
        simple_heading=False,
        resampling_time_range=(0.0, 0.0),
        ranges=TargetPalletCommandCfg.Ranges(
            heading=(-math.pi, math.pi)),
        debug_vis=True,
    )


@configclass
class EventCfg:
    """Randomization configuration for the task."""
    # startup_state = EventTerm(
    #     func=mdp.reset_root_state_pallet_target,
    #     mode="startup",
    #     params={
    #         "pallet_cfg": SceneEntityCfg(name="pallet_scene"),
    #         "target_cfg": SceneEntityCfg(name="target")
    #     },
    # )
    reset_forklift = EventTerm(
        func=mdp.reset_root_state_forklift,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name="forklift"),
        },
    )
    reset_pallet = EventTerm(
        func=mdp.reset_root_state_pallet_target,
        mode="reset",
        params={
            "pallet_cfg": SceneEntityCfg(name="pallet_scene"),
            "target_cfg": SceneEntityCfg(name="target")
        },
    )


# @configclass
# class CurriculumCfg:
#     """ Curriculum configuration for the task. """
#     target_distance = CurrTerm(func=mdp.goal_distance_curriculum)


@configclass
class ForkliftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the rover environment."""

    # Create scene
    scene: ForkliftSceneCfg = ForkliftSceneCfg(
        num_envs=128, env_spacing=10.0, replicate_physics=False)

    # Setup PhysX Settings
    sim: SimCfg = SimCfg(
        physx=PhysxCfg(
            enable_stabilization=True,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**18,
            gpu_found_lost_pairs_capacity=2**20,
            gpu_found_lost_aggregate_pairs_capacity=2**20,  # 2**21,
            gpu_total_aggregate_pairs_capacity=2**20,   # 2**13,
            gpu_max_soft_body_contacts=2**20,
            gpu_max_particle_contacts=1048576,
            gpu_heap_capacity=2**26,
            gpu_temp_buffer_capacity=2**24,
            gpu_max_num_partitions=8,
            gpu_collision_stack_size=2**28,
            friction_correlation_distance=0.025,
            friction_offset_threshold=0.04,
            bounce_threshold_velocity=2.0,
        )
    )
    # print("ForkliftEnvCfg")
    # terrain: TerrainImporterCfg = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="average",
    #         restitution_combine_mode="average",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )

    # Basic Settings
    observations: ObservationCfg = ObservationCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    # MDP Settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.dt = 1 / 30.0
        self.decimation = 6
        self.episode_length_s = 150
        self.viewer.eye = (-30.0, -30.0, 10.0)
        self.viewer.lookat = (-13.0, -13.0, 0.0)


    #     # update sensor periods
    #     if self.scene.height_scanner is not None:
    #         self.scene.height_scanner.update_period = self.sim.dt * self.decimation
    #     if self.scene.contact_sensor is not None:
    #         self.scene.contact_sensor.update_period = self.sim.dt * self.decimation
