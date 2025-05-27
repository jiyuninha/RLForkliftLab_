from typing import Sequence

import random
import math
import torch

from pxr import Sdf, Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema, Tf
import omni.usd
import isaaclab.sim as sim_utils
from isaacsim.core.prims import XFormPrim

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

# Import Pallet Scene 


# Visualization Marker Configuration
SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)
            ),
        ),
    }
)

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg


# Helper Functions (moved outside the class)
def quat_from_yaw():
    """Generate a random quaternion based on a yaw angle between -90° and 90°."""
    y = random.uniform(-math.pi/2, math.pi/2)
    w = math.cos(y / 2)
    z = math.sin(y / 2)
    return Gf.Quatd(w, Gf.Vec3d(0, 0, z))

def cal(distance, yaw):
    """
    Calculate the x and y offsets given a distance and a yaw angle (in degrees).
    """
    dx = math.sin(math.radians(yaw)) * distance
    dy = math.cos(math.radians(180 - yaw)) * distance
    return (dx, dy)

def decide(s0, s1, distance, yaw, pallet_offset=False):
    """
    s0: Forklift's (x, y) position.
    s1: Pallet's (x, y) position.
    
    Calculate two candidate goal points based on the pallet's yaw and return the one
    that is closer to the forklift.
    """
    if pallet_offset:
        distance += 0.4
    a = cal(distance, yaw)
    goal_point1 = (s1[0] - a[0], s1[1] - a[1])
    goal_point2 = (s1[0] + a[0], s1[1] + a[1])
    distance1 = math.dist(goal_point1, s0)
    distance2 = math.dist(goal_point2, s0)
    return goal_point1 if distance1 < distance2 else goal_point2

def quat2yaw(quat):
    """
    Convert a Gf.Quatd quaternion to a yaw angle (in degrees).
    """
    rotation = Gf.Rotation(quat)
    euler_angles = rotation.Decompose(
        Gf.Vec3d(1, 0, 0),
        Gf.Vec3d(0, 1, 0),
        Gf.Vec3d(0, 0, 1)
    )
    return euler_angles[2]  # Z-axis yaw

class TargetPalletCommand(CommandTerm):
    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.forklift: Articulation = env.scene[cfg.asset_name]
        self.stage = omni.usd.get_context().get_stage()
        self.pallet_prim_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/Pallet")

        self.target_distance = 2.0
        self._device = self.forklift.data.device

        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self._device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self._device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.pos_command_b, self.heading_command_b.unsqueeze(1)), dim=1)

    def _update_metrics(self):
        self.metrics["error_pos"] = torch.norm(
            self.pos_command_w - self.forklift.data.root_link_pos_w[:, :3], dim=1)
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.forklift.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        # Step 1: 환경별 pallet 위치와 orientation 무작위 설정
        with Sdf.ChangeBlock():
            for i, path in enumerate(self.pallet_prim_paths):
                pallet_prim = self.stage.GetPrimAtPath(path)
                
                if not pallet_prim:
                    continue
                '''# 팔레트 Xform 객체 얻기 (변환 적용용)
                palletXform = UsdGeom.Xform(pallet_prim)
                
                # 위치 속성 읽거나 생성 후 설정
                translate_attr = pallet_prim.GetAttribute("xformOp:translate")
                
                if translate_attr is None:
                    translate_attr = pallet_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
                translate_attr.Set(rand_pos)
                
                # 방향 속성 읽거나 생성 후 설정
                orient_attr = pallet_prim.GetAttribute("xformOp:orient")
                if orient_attr is None:
                    orient_attr = pallet_prim.CreateAttribute("xformOp:orient", Sdf.ValueTypeNames.Quatd)
                orient_attr.Set(rand_orient)'''
                
                '''# 강체(Rigid Body)로 만들기
                UsdPhysics.RigidBodyAPI.Apply(pallet_prim)
                
                # 메쉬가 있는 Prim 경로 (예: "/World/pallet/myMesh")
                mesh_prim_path = path + '/Pallet_A1'
                mesh_prim = self.stage.GetPrimAtPath(mesh_prim_path)
                
                collision_api = UsdPhysics.CollisionAPI.Apply(pallet_prim)
                # 충돌 API 적용
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
                #PhysxSchema.PhysxCollisionAPI.Apply(mesh_prim)
                
                # Convex Hull 설정
                attr = mesh_collision_api.CreateApproximationAttr("convexHull")
                
                mass_api = UsdPhysics.MassAPI.Apply(pallet_prim)
                mass_api.CreateMassAttr(100)
                #pallet_prim.SetInstanceable(True)'''
                #UsdPhysics.CollisionAPI.Apply(pallet_prim)
                #PhysxSchema.PhysxContactReportAPI.Apply(pallet_prim)
                #UsdPhysics.RigidBodyAPI.Apply(pallet_prim)
                '''# Define terrain mesh
                mesh_prim = stage.DefinePrim(f"/World/{name}", "Mesh")
                mesh_prim.GetAttribute("points").Set(vertices)
                mesh_prim.GetAttribute("faceVertexIndices").Set(faces.flatten())
                mesh_prim.GetAttribute("faceVertexCounts").Set(np.asarray([3] * faces.shape[0]))  # 3 vertices per face
                
                terrain_prim = XFormPrim(
                    prim_path=f"/World/{name}",
                    name=f'{name}',
                    position=position,
                    orientation=orientation)
                    
                UsdPhysics.CollisionAPI.Apply(terrain_prim.prim)
                physx_collision_api: PhysxSchema._physxSchema.PhysxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(terrain_prim.prim)
                physx_collision_api.GetContactOffsetAttr().Set(0.04)
                physx_collision_api.GetRestOffsetAttr().Set(0.01)
                
                material = PhysicsMaterial(
                    prim_path=f"/World/Materials/{name}",
                    # static_friction=0.1,
                    # dynamic_friction=0.8,
                    static_friction=0.1,
                    dynamic_friction=2,
                    restitution=0.0,
                    )
                material2: PhysxSchema._physxSchema.PhysxMaterialAPI = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
                material2.CreateCompliantContactStiffnessAttr().Set(1000000.0)
                material2.CreateCompliantContactDampingAttr().Set(20000.0)'''
                '''
                mesh_prim_path = path + '/Pallet_A1'
                mesh_prim = self.stage.GetPrimAtPath(mesh_prim_path)
                terrain_prim = XFormPrim(
                    prim_path=mesh_prim_path,
                    )
                #UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
                UsdPhysics.CollisionAPI.Apply(terrain_prim.prim)
                physx_collision_api: PhysxSchema._physxSchema.PhysxCollisionAPI = PhysxSchema.PhysxCollisionAPI.Apply(terrain_prim.prim)
                physx_collision_api.GetContactOffsetAttr().Set(0.04)
                physx_collision_api.GetRestOffsetAttr().Set(0.01)'''
                '''rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(pallet_prim)
                rigid_api.GetSolveContactAttr()
                PhysxSchema.PhysxConvexHullCollisionAPI.Apply(pallet_prim)
                PhysxSchema.PhysxContactReportAPI.Apply(pallet_prim)'''
                
                
                # Iterate descendant prims and add colliders to mesh or primitive types
                for desc_prim in Usd.PrimRange(pallet_prim):
                    '''if desc_prim.IsA(UsdGeom.Gprim):
                        # Add rigidbody properties to the prim
                        if not desc_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(desc_prim)
                        else:
                            rigid_body_api = UsdPhysics.RigidBodyAPI(desc_prim)
                        rigid_body_api.CreateRigidBodyEnabledAttr(True)'''
                            
                    if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
                        # Add collision properties to the mesh (make sure collision is enabled)
                        if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                            collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
                        else:
                            collision_api = UsdPhysics.CollisionAPI(desc_prim)
                        collision_api.CreateCollisionEnabledAttr(True)
                        
                        # Add PhysX collision properties to the mesh (e.g. bouncyness)
                        if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                            physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
                        else:
                            physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
                        physx_collision_api.CreateRestOffsetAttr(0.0)
                        
                    '''# Add mesh specific collision properties only to mesh types
                    if desc_prim.IsA(UsdGeom.Mesh):
                        # Add mesh collision properties to the mesh (e.g. collider aproximation type)
                        if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
                        else:
                            mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
                        #desc_prim.GetAttribute("xformOp:orient").Set('convexHull')'''
                '''omni.kit.commands.execute("AddPhysicsComponentCommand",
                      usd_prim=pallet_prim,
                      component="PhysicsRigidBodyAPI")
                mesh_prim = self.stage.GetPrimAtPath(path+'/Pallet_A1')
                omni.kit.commands.execute("AddPhysicsComponentCommand",
                    usd_prim=mesh_prim,
                    component="PhysicsCollisionAPI")
                utils.setCollider(pallet_prim, approximationShape="sdf")'''

                # 랜덤 위치 생성
                rand_x = random.uniform(-3.0, 3.0)
                rand_y = random.uniform(-3.0, 3.0)
                pos = Gf.Vec3d(rand_x, rand_y, 0.0)
                
                # 랜덤 yaw 쿼터니언 (사용자 정의 함수)
                quat = quat_from_yaw()
                
                # Xform 정의 및 위치/회전 설정
                pallet_xform = UsdGeom.Xform(pallet_prim)
                xform_ops = pallet_xform.GetOrderedXformOps()
                
                translate_op = None
                orient_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        orient_op = op
                # 기존 transform 연산자가 있으면 값만 설정, 없으면 새로 추가
                if translate_op:
                    translate_op.Set(pos)
                else:
                    pallet_xform.AddTranslateOp().Set(pos)
                
                if orient_op:
                    orient_op.Set(quat)
                else:
                    pallet_xform.AddOrientOp().Set(quat)
                

        # Step 2: pallet 기준으로 각 환경의 target point 설정
        for i, path in enumerate(self.pallet_prim_paths):
            pallet_prim = self.stage.GetPrimAtPath(path)
            if not pallet_prim:
                continue

            translate_attr = pallet_prim.GetAttribute("xformOp:translate")
            orient_attr = pallet_prim.GetAttribute("xformOp:orient")
            if translate_attr is None or orient_attr is None:
                continue

            translate = translate_attr.Get()
            orient = orient_attr.Get()
            yaw = quat2yaw(orient)

            s1 = (translate[0], translate[1])  # pallet 위치
            s0 = (0.0, 0.0)  # 기준점 (무시 가능)

            # target 후보 계산
            a = cal(self.target_distance, yaw)
            candidate1 = (s1[0] - a[0], s1[1] - a[1])
            candidate2 = (s1[0] + a[0], s1[1] + a[1])

            selected_xy = random.choice([candidate1, candidate2])
            selected_z = 0.0  # flat ground 가정

            selected_target = torch.tensor([*selected_xy, selected_z], device=self._device)

            if i in env_ids:
                self.pos_command_w[i] = selected_target
                self.heading_command_w[i] = 0.0

    def _update_command(self):
        target_vec = self.pos_command_w - self.forklift.data.root_link_pos_w[:, :3]
        self.pos_command_b = quat_rotate_inverse(yaw_quat(self.forklift.data.root_link_quat_w), target_vec)
        self.heading_command_b = wrap_to_pi(self.heading_command_w - self.forklift.data.heading_w)



if __name__ == "__main__":
    importer = TargetPalletCommand()
    new_targets = importer.sample_new_target()
    print("Target points for each environment:", new_targets)
    pass
