from __future__ import annotations

import argparse
import traceback
from typing import TYPE_CHECKING

import carb
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Forklift and Pallet Scene")

parser.add_argument("--num_envs", type=int, default=64,
                    help="Number of environments to create")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: F401, E402
from isaaclab.assets import (
    Articulation, 
    ArticulationCfg, 
    AssetBaseCfg, 
    RigidObject, 
    RigidObjectCfg,
)# noqa: F401, E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: F401, E402
from isaaclab.sim import SimulationContext  # noqa: F401, E402
from isaaclab.utils import configclass  # noqa: F401, E402

# Get Forklift, pallets
from forklift_envs.assets.forklift_pallet_cfg import FORKLIFT_CFG  # noqa: F401, E402
from forklift_envs.assets.forklift_pallet_cfg import PALLET_CFG  # noqa: F401, E402


if TYPE_CHECKING:
    from forklift_envs.envs.local_navigation.utils.articulation.articulation import ForkliftArticulation
# Here we configure the environment


@configclass
class ForkliftPalletSceneCfg(InteractiveSceneCfg):
    """ Configuration for the forklift and pallet scene """

    # Add ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Add lightsss
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Add the forklift
    forklift: ArticulationCfg = FORKLIFT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Forklift")
    
    # Add the pallet
    pallet: ArticulationCfg = PALLET_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Pallet")

def setup_scene():
    """ Setup the scene """
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    
    # Set Default Camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = ForkliftPalletSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    return sim, scene


'''
Update pallet's position and rotation randomly.
'''
import torch
import numpy as np

def random_quaternion():
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return torch.tensor([qx, qy, qz, qw], dtype=torch.float32)

def random_position(x_range=(0, 2), y_range=(-2, 2)):
    x = np.random.uniform(*x_range)
    y = np.random.uniform(*y_range)
    return torch.tensor([x, y, 0.0], dtype=torch.float32)


def run_simulation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation """
    # Get the forklift
    forklift: ForkliftArticulation = scene["forklift"]
    # Get the pallet
    pallet: RigidObjectCfg = scene["pallet"] 
        
    sim_dt = sim.get_physics_dt()
    count = 0
    num_envs = pallet.data.default_root_state.shape[0]  

    def reset_scene(forklift: ForkliftArticulation, pallet: Articulation, scene: InteractiveScene):
        root_state = forklift.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        forklift.write_root_state_to_sim(root_state)
        
        new_positions = torch.stack([random_position() for _ in range(num_envs)])
        new_rotations = torch.stack([random_quaternion() for _ in range(num_envs)])

        joint_pos, joint_vel = forklift.data.default_joint_pos.clone(), forklift.data.default_joint_vel.clone()
        joint_pos += torch.randn_like(joint_pos) * 0.1
        
        forklift.write_joint_state_to_sim(joint_pos, joint_vel)
        pallet.write_root_pose_to_sim(torch.cat([new_positions, new_rotations], dim=-1))

        scene.reset()
        print("Reset")

    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0

            # Reset the scene
            reset_scene(forklift, pallet, scene)

        # Step the simulation

        velocities = torch.ones_like(forklift.data.default_joint_pos) * 0.1
        # robot.set_joint_effort_target(efforts)
        forklift.set_joint_velocity_target(velocities)

        scene.write_data_to_sim()

        sim.step()

        count += 1

        scene.update(sim_dt)


def main():
    # First we setup the scene
    sim, scene = setup_scene()
    # Then we run the simulation
    run_simulation(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Error in main: {e}")
        carb.log_error(traceback.format_exc())
    finally:
        simulation_app.close()
