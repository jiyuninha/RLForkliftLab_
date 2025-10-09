# RLForkliftLab: RL-Based Navigation for Forklifts in Isaac Lab

**RLForkliftLab** is an **Isaac Lab / Isaac Sim**–based **reinforcement learning (RL)** environment for **forklifts**.  
Keywords: RLForkliftLab, Isaac Lab, Isaac Sim, ROS 2, forklift, PPO, navigation.

<p align="left">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-22.04-important">
  <img alt="GPU" src="https://img.shields.io/badge/NVIDIA-GPU%20required-lightgrey">
    <a href="https://github.com/NVIDIA-Omniverse/IsaacLab">
    <img alt="IsaacLab" src="https://img.shields.io/badge/IsaacLab-2.1.0-brightgreen">
  </a>
  <a href="https://developer.nvidia.com/isaac-sim">
    <img alt="IsaacSim" src="https://img.shields.io/badge/IsaacSim-4.5.0-brightgreen">
  </a>
</p>

## Features
- Forklift kinematics/steering model and Isaac Lab **task** template
- Example **training/eval** scripts and a custom **env** (PPO/SAC-ready)
- Quick demo launcher (`isaaclab.sh -p ...`)

---

## Prerequisites
- **Ubuntu 22.04**, **ROS 2 Humble**
- **Docker**, **Docker Compose v2**
- **NVIDIA Container Toolkit** (GPU required)
- (Recommended) **NVIDIA Driver 535+**

> When using GPU with GUI, you may need to configure X permissions/display settings.

---

1. Set environment
![[Check] RL Env](figure/test2.png)
2. Check articulation operation
![[Check] Forklift articulation](figure/test6.gif)
3. Set train code
![[Check] Train pipeline](figure/test9.gif)
