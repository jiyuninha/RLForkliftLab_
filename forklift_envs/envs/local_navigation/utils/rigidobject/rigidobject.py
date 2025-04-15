"""
rigidobject.py

PalletRigidObject와 그 설정 클래스(PalletRigidObjectCfg)를 정의합니다.
PalletRigidObject는 초기화 과정에서 USD 프라임에 'USD RigidBodyAPI'와
충돌 데이터를 구성할 수 있는 collision prim이 적용되어 있는지 확인하고,
없다면 동적으로 생성한 후, 자신의 환경(env_idx)에 맞게 contact report 대상(target)을 등록합니다.
"""

from dataclasses import dataclass, field
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils import configclass

# USD 및 PhysX 관련 모듈 임포트
from pxr import PhysxSchema, Usd, UsdGeom
from omni.usd import get_context  # 현재 스테이지를 가져오기 위해 사용

from omni.usd import get_context
import sys


@configclass
class PalletRigidObjectCfg(RigidObjectCfg):
    """
    PalletRigidObjectCfg는 팔레트 객체의 설정을 담습니다.
    RigidObjectCfg를 상속받아 기본 스폰 설정 외에 추가 파라미터를 확장할 수 있습니다.
    
    env_idx: 해당 pallet이 속한 환경 인덱스 (예: 0, 1, 2, …)
    """
    env_idx: int = 0

class PalletRigidObject(RigidObject):
    """
    PalletRigidObject는 PalletRigidObjectCfg 설정을 기반으로 생성되는
    팔레트 객체입니다. 초기화 시, 해당 USD 프라임에 RigidBodyAPI와 충돌 데이터를 구성할 수 있는
    collision prim이 적용되어 있는지 확인하고, 없으면 동적으로 추가한 후,
    자신의 env_idx에 맞춰 contact report 대상(target)을 등록합니다.
    """
    def __init__(self, cfg: PalletRigidObjectCfg):
        super().__init__(cfg)

    def _initialize_impl(self):
        """
        초기화 구현을 오버라이드하여, 해당 USD 프라임에 'USD RigidBodyAPI'가 적용되었는지 확인하고,
        미적용 상태면 이를 추가한 후, collision prim(충돌 geometry)이 존재하는지 확인하여 없으면 생성합니다.
        그 후, 자신의 환경 인덱스(env_idx)를 기반으로 contact report 대상(target)을 등록합니다.
        """
        # 현재 스테이지 가져오기
        stage = get_context().get_stage()
        print("test1")
        self.print_all_prims()
        prim = stage.GetPrimAtPath(self.cfg.prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Prim at path {self.cfg.prim_path} not found.")

        # RigidBodyAPI 적용 (없으면 적용)
        if not prim.HasAPI(PhysxSchema.RigidBodyAPI):
            PhysxSchema.RigidBodyAPI.Apply(prim)

        env_name = f"/World/envs/env_{env_idx}"  # ✅ 환경 인덱스를 환경 이름으로 변환
        contact_api.CreateReportPairsRel().AddTarget(f"{env_name}/Pallet")



        # collision geometry 확인: prim 아래에 "Collision" 자식 prim을 확인
        collision_path = prim.GetPath().AppendChild("Collision")
        collision_prim = stage.GetPrimAtPath(collision_path)
        if not collision_prim or not collision_prim.IsValid():
            # collision prim이 없으므로 생성 (여기서는 Cube 형태로 생성)
            collision_prim = stage.DefinePrim(collision_path, "Cube")
            # 생성된 collision prim에 PhysxCollisionAPI 적용
            PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
            # USD Geometry API를 사용하여 충돌체 속성 설정 (예: 크기)
            cube = UsdGeom.Cube(collision_prim)
            cube.CreateSizeAttr(1.0)  # 실제 팔레트 크기에 맞게 조정 필요

        # 환경 인덱스(env_idx)는 cfg에 정의되어 있다고 가정합니다.
        env_idx = self.cfg.env_idx

        # contact report 대상 등록
        # 이 코드는 pallet prim의 contact report를 forklift의 contact sensor가 올바른 경로를 참조하도록 하기 위한 것입니다.
        contact_api = PhysxSchema.PhysxContactReportAPI.Get(stage, prim)
        # 각 환경의 pallet 경로를 "/World/envs/env_{env_idx}/Pallet" 형태로 지정
        contact_api.CreateReportPairsRel().AddTarget(f"/World/envs/env_{env_idx}/Pallet")

        # 이후 부모 클래스의 초기화 로직 실행 (여기서 _data 속성이 생성됨)
        super()._initialize_impl()

    def print_all_prims(self):
        """
        현재 USD 스테이지에 있는 모든 prim의 경로를 터미널에 출력합니다.
        """
        # 현재 스테이지 가져오기
        print("print_all_prims")
        stage = get_context().get_stage()
        
        # 스테이지 내 모든 prim 순회 및 출력
        print("Stage에 있는 모든 prims:")
        for prim in stage.Traverse():
            print(prim.GetPath())
