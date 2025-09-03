# examples/test_code/check_usd_info.py
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True})   # Kit 런타임 시작 (GUI 필요 없으면 headless=True)

from pxr import Usd, UsdPhysics

USD_PATH = "/workspace/isaac_forklift/forklift_envs/assets/robots/forklift_b/forklift_b.usd"  # 경로 맞게 수정

# USD 파일 열기 (스테이지 로드)
stage = Usd.Stage.Open(USD_PATH)
root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
print("[root prim]:", root.GetPath())

# RigidBody / Mass 정보 훑기
print("\n[Rigid bodies / Mass prims]")
body_paths = []
for prim in stage.Traverse():
    path = prim.GetPath()
    rb_api = UsdPhysics.RigidBodyAPI.Get(stage, path)
    mass_api = UsdPhysics.MassAPI.Get(stage, path)

    if rb_api or mass_api:
        body_paths.append(path.pathString)
        mass = None
        if mass_api:
            mass_attr = mass_api.GetMassAttr()
            if mass_attr and mass_attr.HasAuthoredValueOpinion():
                mass = mass_attr.Get()
        print(f"  - {path}   mass={mass}")

print("\n[body_paths]")
for p in body_paths:
    print(" ", p)

simulation_app.close()
