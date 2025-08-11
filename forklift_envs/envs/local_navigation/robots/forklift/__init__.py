import os
import gymnasium as gym

# custom_env 모듈에서 에이전트 생성 함수와 환경 설정 클래스를 임포트합니다.
from forklift_envs.envs.local_navigation.skrl import get_agent
from . import env_cfg

# Forklift 환경 등록
gym.register(
    id="ForkliftEnv-v0",
    entry_point="forklift_envs.envs.local_navigation.entrypoints:ForkliftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.TriCycleForkliftEnvCfg,
        "best_model_path": f"{os.path.dirname(__file__)}/policies/_test_best.pt",
        "get_agent_fn": get_agent,
        "skrl_cfgs": {
            "PPO": f"{os.path.dirname(__file__)}/../../skrl/configs/forklift_ppo.yaml",
        }
    }
)

# # Forklift 카메라 환경 등록
# gym.register(
#     id="ForkliftEnvCamera-v0",
#     entry_point="custom_env.envs.forklift.entrypoints:ForkliftEnvCamera",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": env_cfg.ForkliftEnvCfg,
#         "best_model_path": f"{os.path.dirname(__file__)}/policies/forklift_camera_best_agent.pt",
#         "get_agent_fn": get_agent,
#     }
# )
