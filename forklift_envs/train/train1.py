#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic training process for the forklift environment.

This script loads the training environment from forklift_empty_envs2.py,
then runs a basic training loop using random actions.
"""

import argparse
import os
import random
import torch
import gymnasium as gym

from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from forklift_envs.assets.forklift_empty_envs2 import ForkliftSceneCfg  # 환경 구성 클래스 불러오기

# --- Utility functions ---

def random_action(joint_shape):
    """
    주어진 joint 상태(shape)에 맞춰 임의의 액션(노이즈)을 생성.
    실제 학습 알고리즘에 맞게 수정 필요.
    """
    return torch.randn(joint_shape) * 0.1

def compute_reward(root_state):
    """
    간단한 보상 함수 예시.
    여기서는 forklift의 root position이 (0,0,0)에 가까울수록 (norm이 작을수록) 높은 보상을 준다고 가정.
    (실제 환경에 맞게 수정 필요)
    """
    # root_state의 첫 3원소를 위치로 간주하여 norm을 계산
    pos = root_state[:3]
    reward = -torch.norm(pos).item()
    return reward

def train_agent(sim, scene, num_episodes=10, episode_length=1000):
    """
    기본 학습 루프 함수.
    - 매 에피소드마다 환경을 리셋하고,
    - 일정 시간 동안 임의의 액션을 적용하며 보상을 계산한 뒤 에피소드 총 보상을 출력합니다.
    """
    # InteractiveScene에서 forklift 자산에 접근 (예: 키 "forklift")
    forklift = scene["forklift"]
    
    for ep in range(num_episodes):
        sim.reset()  # 에피소드마다 환경 리셋
        total_reward = 0.0
        # 초기 관찰값: forklift의 default root state (예시)
        obs = forklift.data.default_root_state.clone()
        
        for step in range(episode_length):
            # 현재 joint 상태의 shape에 맞춰 임의의 액션 생성
            action = random_action(forklift.data.default_joint_pos.shape)
            # forklift에 액션 적용 (여기서는 joint velocity target로 사용)
            forklift.set_joint_velocity_target(action)
            sim.step()  # 시뮬레이션 스텝 진행
            
            # 다음 상태 관찰 (예시: default root state)
            next_obs = forklift.data.default_root_state.clone()
            reward = compute_reward(next_obs)
            total_reward += reward
            obs = next_obs
        
        print(f"Episode {ep}: Total Reward = {total_reward}")

# --- Main function ---

def main():
    # 시뮬레이션 설정 및 컨텍스트 생성
    sim_cfg = sim_utils.SimulationCfg(device="cpu")  # 필요에 따라 args.device 등으로 수정
    sim = SimulationContext(sim_cfg)
    
    # 카메라 설정
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    
    # 환경 구성 불러오기
    scene_cfg = ForkliftSceneCfg(num_envs=64, env_spacing=10)
    print("[INFO]: Scene is being loaded...")
    scene = InteractiveScene(scene_cfg)
    print("[INFO]: Scene successfully loaded!")
    
    sim.reset()
    print("[INFO]: Setup complete!!!")
    
    # 기본 학습 루프 실행
    train_agent(sim, scene, num_episodes=10, episode_length=1000)
    
    # 학습 종료 후 추가 시뮬레이션 루프 (원하는 경우)
    while simulation_app.is_running():
        sim.step()

if __name__ == "__main__":
    main()
