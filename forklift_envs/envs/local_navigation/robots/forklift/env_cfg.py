from __future__ import annotations

from isaaclab.utils import configclass

import forklift_envs.mdp as mdp
from forklift_envs.assets.robots.forklift import FORKLIFT_CFG
from forklift_envs.envs.local_navigation.forklift_env_cfg import ForkliftEnvCfg


@configclass
class TriCycleForkliftEnvCfg(ForkliftEnvCfg):
    """Configuration for the Forklift environment."""

    def __post_init__(self):
        super().__post_init__()

        # Define robot
        self.scene.forklift = FORKLIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Forklift")

        # Define parameters for the Ackermann kinematics.s
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="forklift",
            wheelbase_length=1.5,
            middle_wheel_distance=0.8,
            rear_and_front_wheel_distance=1.6,
            wheel_radius=0.2,
            min_steering_radius=0.1,
            steering_joint_names=[".*back_wheel_swivel"],
            drive_joint_names=[".*back_wheel_drive"],
            offset=-0.0135
        )
