import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from forklift_envs.envs.local_navigation.utils.articulation.articulation import ForkliftArticulation

# _AAU_ROVER_PATH = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "assets", "rover", "rover_instance.usd"
# )
# _AAU_ROVER_PATH = "http://localhost:8080/omniverse://127.0.0.1/Projects/simple_instanceable_new2/rover_instance.usd"
# _AAU_ROVER_PATH = "http://localhost:8080/omniverse://127.0.0.1/Projects/simplified9.usd"
# _AAU_ROVER_SIMPLE_PATH =
# "http://localhost:8080/omniverse://127.0.0.1/Projects/simple_instanceable_new/rover_instance.usd"
FORKLIFT_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "forklift_b", "forklift_b.usd")

STEER_JOINT = ['back_wheel_swivel']
DRIVE_JOINT = ['back_wheel_drive']
PASSIVE_JOINTS = ['front_right_roller', 'back_left_roller', 'back_right_roller', 'front_left_roller'] 
LIFT_JOINT = ["lift_joint"]

FORKLIFT_CFG = ArticulationCfg(
    class_type=ForkliftArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=FORKLIFT_USD_PATH,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1.5,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "steer_actuator": ImplicitActuatorCfg(
            #class_type=ImplicitActuatorCfg,
            joint_names_expr="|".join(STEER_JOINT),
            effort_limit=20.0,
            velocity_limit=3.0,
            stiffness=100.0,
            damping=1000.0,
        ),
        "drive_actuator": ImplicitActuatorCfg(
            joint_names_expr="|".join(DRIVE_JOINT),
            velocity_limit=6,
            effort_limit=12,
            stiffness=100.0,
            damping=4000.0,
        ),
        "passive_actuator": ImplicitActuatorCfg(
            joint_names_expr="|".join(PASSIVE_JOINTS),
            effort_limit=0,
            velocity_limit=12.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "lift_actuator": ImplicitActuatorCfg(
            #class_type=ImplicitActuatorCfg,
            joint_names_expr="|".join(LIFT_JOINT),
            effort_limit=200.0,
            velocity_limit=5.0,
            stiffness=8000.0,
            damping=1000.0
        ),
    },
)
