import os
import random

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import   RigidObject, RigidObjectCfg

from forklift_envs.envs.local_navigation.utils.articulation.articulation import ForkliftArticulation
from forklift_envs.envs.local_navigation.utils.rigidobject.rigidobject import PalletRigidObject
# Pallet Config

_PALLET_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "pallets", "Pallet_A1.usd")

PALLET_CFG = RigidObjectCfg(
    class_type=PalletRigidObject,
    spawn=sim_utils.UsdFileCfg(
        usd_path=_PALLET_USD_PATH,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), # random position, pose
    ),
)


# Forklift Config

_FORKLIFT_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "forklift_b", "forklift_b.usd")

FORKLIFT_CFG = ArticulationCfg(
    class_type=ForkliftArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FORKLIFT_USD_PATH,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.01), # 4mm, 1cm
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1.5,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=50,
            solver_velocity_iteration_count=10,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-1.5, 0.0, 0.0),
        joint_pos={".*lift_joint": 0.0},
        joint_vel={".*lift_joint": 0.0, ".*back_wheel_drive": 0.0},
    ),
    actuators={
        "steering_actuator": IdealPDActuatorCfg(
            joint_names_expr=[".*back_wheel_swivel"],
            velocity_limit=6,
            effort_limit=100,
            stiffness=8000.0,
            damping=1000.0,
        ),
        "drive_actuator": IdealPDActuatorCfg(
            joint_names_expr=[".*back_wheel_drive"],
            velocity_limit=6,
            effort_limit=100,
            stiffness=50.0,
            damping=4000.0,
        ),
        "passive_actuator": IdealPDActuatorCfg(
            joint_names_expr=[".*front_right_roller", ".*back_left_roller", ".*back_right_roller", ".*front_left_roller"],
            velocity_limit=6,
            effort_limit=0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
