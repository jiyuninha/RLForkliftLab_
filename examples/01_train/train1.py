import argparse 
import math 
import os
import random
import sys
from datetime import datetime

import carb
import gymnasium as gym 
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Isaac Lab: Omniverse Robotics Environments!")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ForkliftEnv-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--agent", type=str, default="PPO", help="Name of the agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.skrl import SkrlVecEnvWrapper 

# Omniverse의 설정을 가져와 ray tracing 관련 설정을 조정
carb_settings = carb.settings.get_settings()
carb_settings.set_bool(
    "rtx/raytracing/cached/enabled",
    False,
)
carb_settings.set_int(
    "rtx/descriptorSets",
    8192,
)

# Isaac Lab의 강화학습 환경 및 유틸리티 모듈 임포트
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_pickle, dump_yaml  # noqa: E402

# 로그 디렉토리 및 설정 구성/저장
def log_setup(experiment_cfg, env_cfg, agent):
    """
    Setup the logging for the experiment.

    Note:
        Copied from the Isaac Lab framework.
    """
    # 실험 로그의 기본 경로를 logs/skrl/[experiment directory]로 지정
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # 현재 시간을 기반으로 로그 디렉토리 이름 생성
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir = f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'

    log_dir += f"_{agent}"

    # 에이전트 설정에 로그 디렉토리와 실험 이름 반영
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # 
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)
    return log_dir

# 비디오 녹화 설정 함수: 비디오 기록이 활성화된 경우 환경에 RecordVideo 래퍼를 적용
def video_record(
    env: ManagerBasedRLEnv, log_dir: str, video: bool, video_length: int, video_interval: int
) -> ManagerBasedRLEnv:
    """
    Function to check and setup video recording.

    Note:
        Copied from the Isaac Lab framework.

    Args:
        env (ManagerBasedRLEnv): The environment.
        log_dir (str): The log directory.
        video (bool): Whether or not to record videos.
        video_length (int): The length of the video (in steps).
        video_interval (int): The interval between video recordings (in steps).

    Returns:
        ManagerBasedRLEnv: The environment.
    """
    # 비디오 옵션이 True면,
    if video:
        # 비디오 녹화 설정을 위한 인자들을 딕셔너리로 구성
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"), # 비디오 저장 폴더
            "step_trigger": lambda step: step % video_interval == 0, # 녹화 시작 트리거 
            "video_length": video_length, # 비디오 녹화 길이
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4) # 설정 내용 출력 
        return gym.wrappers.RecordVideo(env, **video_kwargs) 

    return env

# 테스크 환경 설정을 파싱하는 함수
from isaaclab.sim import SimulationCfg

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
# SKRL의 순차적 학습 트레이너 클래스 임포트
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
# 시드 설정 함수
from skrl.utils import set_seed  # noqa: E402, F401

# 로버 네비게이션 관련 모듈 임포트
import forklift_envs.envs.local_navigation.robots.forklift  # noqa: E402, F401
# Import agents
# 지정된 에이전트 이름에 따라 에이전트를 생성하는 함수
from forklift_envs.envs.local_navigation.skrl import get_agent  # noqa: E402 
# SKRL 에이전트 설정을 파싱하는 함수
from forklift_envs.utils.config import parse_skrl_cfg  # noqa: E402

# train 함수:전체 학습 프로세스를 구성하는 메인 함수
def train():
    # 시드 값 설정: 커멘드라인에서 주어지면 사용, 없으면 무작위로 생성
    args_cli_seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    # 테스크 이름, 디바이스, 환경 개수를 기반으로 환경 설정 파싱
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0" if not args_cli.cpu else "cpu", num_envs=args_cli.num_envs)
    # 태스크와 에이전트 이름을 결합하여 SKRL 실험 설정 파싱
    experiment_cfg = parse_skrl_cfg(args_cli.task + f"_{args_cli.agent}")
    log_dir = log_setup(experiment_cfg, env_cfg, args_cli.agent)

#     # Create the environment
#     # 환경 생성: render_mode는 비디오 기록 옵션에 따라 설정
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, viewport=args_cli.video, render_mode=render_mode)
#     # Check if video recording is enabled
#     # 비디오 기록이 활성화되어 있다면, video_record 함수를 통해 환경을 래핑
    env = video_record(env, log_dir, args_cli.video, args_cli.video_length, args_cli.video_interval)
#     # SKRL 벡터 환경 래퍼로 감싸서 병렬 실행 및 통합 인터페이스 제공
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    # print("SkrlVecEnvWrapper : ", env)
#     # 재현성 확보를 위해 시드 설정
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

#     # Get the observation and action spaces
#     # 환경에서 관측 및 행동 차원 정보를 가져옴
    num_obs = env.unwrapped.observation_manager.group_obs_dim["policy"][0]
    num_actions = env.unwrapped.action_manager.action_term_dim[0]
#     # OpenAI Gym의 Box 스페이스를 사용하여 관측 공간 정의 -> observation space
    observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
#     # 행동 공간 정의 (-1과 1 사이의 값)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))
#     # 관측 공간, 행동 공간, 환경 개수 및 관련 정보 출력
    print(f'Observation space: {observation_space.shape}')
    print(f'Action space: {action_space.shape}')
    print(f'num envs: {env.num_envs}')
    print(f'env obs space: {env.observation_space}')
    print(f'env action space: {env.action_space}')
    # exit()
    # 실험 설정에서 트레이너 설정 부분
    trainer_cfg = experiment_cfg["trainer"]
    
    # 지정된 에이전트 이름, 환경, 관측 및 행동 공간, 실험 설정을 기반으로 에이전트 생성
    agent = get_agent(args_cli.agent, env, observation_space, action_space, experiment_cfg, conv=True)
    # SequentialTrainer를 생성하여 에이전트와 환경을 인자로 전달, 학습 준비
    trainer = SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)
    # 학습 시작
    trainer.train()

    # 학습 종료 후, 환경 및 시뮬레이션 앱 종료로 자원 해제
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    train()
