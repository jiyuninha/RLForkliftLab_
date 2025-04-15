import os
import gymnasium as gym

# custom_env 모듈에서 에이전트 생성 함수와 환경 설정 클래스를 임포트합니다.
from forklift_envs.envs.local_navigation.skrl import get_agent
from . import env_cfg

# Forklift 환경 등록
gym.register(
    id="ForkliftEnv-v0",
    # entry_point: 환경 객체를 생성할 때 사용할 실제 클래스의 경로를 지정.
    entry_point="forklift_envs.envs.local_navigation.entrypoints:ForkliftEnv",
    # Gym에서 환경 생성 시 기본적으로 수행하는 환경 스펙 검사(Checker)를 비활성화.
    # 환경이 Gym의 표준에 완전히 부합하지 않더라도 등록할 수 있도록 허용.
    disable_env_checker=True,
    kwargs={
        # 환경 생성 시 사용할 설정 클래스를 지정.
        # 'env_cfg.AAURoverEnvCfg'는 해당 환경의 초기 설정, 구성 정보를 제공하는 역할.
        "env_cfg_entry_point": env_cfg.TriCycleForkliftEnvCfg,
        # 학습 과정 중 성능이 가장 좋았던 모델(체크포인트)을 이 경로에 저장하거나 불러오기 위해 사용
        "best_model_path": f"{os.path.dirname(__file__)}/policies/forklift_best_agent.pt",
        # 에이전트를 생성하기 위한 함수(get_agent)를 전달.
        # 환경에 맞는 강화학습 에이전트를 생성할 때 사용됨.
        "get_agent_fn": get_agent,  
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
