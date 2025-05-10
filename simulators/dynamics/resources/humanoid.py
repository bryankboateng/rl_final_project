# Just some humanoid dynamics that corresponds with the failure states.
# angles = radians; time = seconds; distances = meters

# Conditions that can cause failure:

# Format:   Link Name       Conditional     Threshold

# Pelvis Height is basically on the floor. 
# 1.        "Root" Z-Coord      <               0.5

# Rotation and Pitch are calculated from the Sim quaternion
# Robot is rotated too far in any direction:
# 2.        abs("Root" Roll)    >               0.7
# 3.        abs("Root" Pitch)   >               0.7

#           Left_Foot_Grounded  and Right_Foot_Grounded are true when 
#           sim detects feet collision with ground plane.
# -         Airborn      !=   (Left_Foot_Grounded | Right_Foot_Grounded)
# -         Start Timer when airborn becomes true.
# -         Air_Time     = Time in air

# Robot is airborn for too long and pelvis is too low:
# 4.        (Air_Time  >  0.3) & ("Root" Z-Coord < 0.7)

# Any joint with limits are out of bounds:
# 5.         Knee_Joint_State   <       -3.14 + 0.05
# 6.         Knee_Joint_State   >       0.0  - 0.05
# 7.         Elbow_Joint_State  <       0.0  - 0.05
# 8.         Elbow_Joint_State  >       3.14 + 0.05

import numpy as np
import pybullet as p
import os
import math
# from .utils import *
from scipy.spatial.transform import Rotation
import time
import pybullet_data


class Humanoid:
    def __init__(self, client, height=1.5, orientation=None, env_type=None, payload_max=0, **kwargs):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client = client
        robot_path = "/home/bb/Desktop/rl_final_project/simulators/dynamics/resources/humanoid/humanoid.urdf"
        self.id = p.loadURDF(robot_path, basePosition=[0, 0, 1], baseOrientation=p.getQuaternionFromEuler([1.57, 0, 0]), useFixedBase=False)
        self.type = "sim"
        self.dim_x = kwargs.get("dim_x", 17)
        self.action_type = kwargs.get("action_type", "center_sampling")
        self.center = kwargs.get("action_center", None)
        self.target_list = kwargs.get("target_list", [])
        self.safety_list = kwargs.get("safety_list", [])

        self.joint_index = self.make_joint_list()
        #assert len(self.joint_index) == 10, f"Expected 10 joints, got {len(self.joint_index)}. Check URDF names."

        self.feet_index = self.make_feet_list()
        #self.controllable_joints = self.get_controllable_joints()
        self.last_contact = np.zeros(2, dtype=bool)  # [left_foot_contact, right_foot_contact]
        self.feet_air_time = np.zeros(2, dtype=float)
        self.plane = p.loadURDF("plane.urdf")
        self.spherical_list = [b'left_ankle', b'right_ankle', b'left_hip', b'right_hip', b'chest']
        self.spherical_joints = [idx for idx in self.joint_index if p.getJointInfo(self.id, idx)[2] == p.JOINT_SPHERICAL and p.getJointInfo(self.id, idx)[1] in self.spherical_list]
        self.revolute_joints = [idx for idx in self.joint_index if p.getJointInfo(self.id, idx)[2] == p.JOINT_REVOLUTE]
        
        # print(f"[Humanoid] Loaded humanoid with ID {self.id}, controllable joints: {self.joint_index}")
        # print(f"[Humanoid] Spherical joints: {self.spherical_joints}")
        # print(f"[Humanoid] Revolute joints: {self.revolute_joints}")


    def find_plane(self):
        for i in range(p.getNumBodies(self.client)):
            name = p.getBodyInfo(i, self.client)[1].decode("ascii")
            if "plane" in name:
                return i
        return -1
    
    def split_action(self, full_action):
        """
        Splits a flat action list into:
        - a list of revolute joint actions
        - a list of (roll, pitch, yaw) tuples for each spherical joint
        """
 
        assert len(full_action) == 19, f"Expected 16 elements, got {len(full_action)}"
        revolute = full_action[:4]
        spherical = [
            tuple(full_action[4:7]),   # right_hip
            tuple(full_action[7:10]),  # right_ankle
            tuple(full_action[10:13]), # left_hip
            tuple(full_action[13:16]),  # left_ankle
            tuple(full_action[16:19])  # chest
        ]
        return revolute, spherical

    def merge_action(self, revolute, spherical):
        """
        Merges a list of revolute joint values and a list of (roll, pitch, yaw) spherical joint tuples
        back into a flat action list
        """
        assert len(revolute) == 4, f"Expected 4 revolute joints, got {len(revolute)}"
        assert len(spherical) == 5, f"Expected 4 spherical joints, got {len(spherical)}"
        flat_spherical = [angle for triple in spherical for angle in triple]
        return revolute + flat_spherical
    

    def make_joint_list(self):
        # removed shoulder joints
        # b'right_shoulder', b'left_shoulder',
        joint_names = [
            b'right_hip', b'right_knee', b'right_ankle',
            b'left_hip', b'left_knee', b'left_ankle',
            b'right_elbow',
            b'left_elbow', b'chest',
        ]
        #[0-3, 4, 5-8, 9-12, 13]
        joint_list = []
        all_joints = p.getNumJoints(self.id, self.client)
        #print("all_joints", all_joints)

        for jname in joint_names:
            for i in range(all_joints):
                name = p.getJointInfo(self.id, i, self.client)[1]
                #print("name", name, "i", i)
                if name == jname:
                    joint_list.append(i)

        # if len(joint_list) != 10:
        #     print(f"[Humanoid] ⚠️ Joint list mismatch: expected 10, got {len(joint_list)}. List: {joint_list}")
        # print(f"[Humanoid] Joint list: {joint_list}")
        return joint_list


    def make_feet_list(self):
        foot_names = [b'right_ankle', b'left_ankle']
        foot_list = []
        for fname in foot_names:
            for i in range(p.getNumJoints(self.id, self.client)):
                name = p.getJointInfo(self.id, i, self.client)[1]
                if name == fname:
                    foot_list.append(i)
        return foot_list

    # def reset(self, position, velocity=None):
    #     for idx in self.joint_index:
    #         p.resetJointState(self.id, idx, 0.0, 0.0, physicsClientId=self.client)

    def apply_action(self, action):
        if self.action_type == "increment":
            ValueError(f"Unsupported action_type: {self.action_type}")
        elif self.action_type == "center_sampling":
            targets = np.array(self.center) + np.array(action)
            revolute_target, spherical_target = self.split_action(targets)
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
        self._apply_joint_targets2(revolute_target, spherical_target, use_gains=False)

    def apply_position(self, position=None, default=False):
        
        if default:
            position = np.array(self.center) if not isinstance(self.center, np.ndarray) else self.center
            
        revolute_target, spherical_target = self.split_action(position)
        self.apply_position2(revolute_target, spherical_target)

    def apply_position2(self, joint_angles, spherical_angles):
        self._apply_joint_targets2(joint_angles, spherical_angles, use_gains=True)

    def _apply_joint_targets2(self, revolute_targets=None, spherical_targets=None, use_gains=False):
        """
        Apply separate controls for revolute and spherical joints with clipping and optional gains.
        """
        # === Apply revolute joint controls ===
        if revolute_targets is not None:
            for i, idx in enumerate(self.revolute_joints):
                info = p.getJointInfo(self.id, idx, physicsClientId=self.client)
                lo, hi = info[8], info[9]
                force = info[10]
                vel = info[11]
                target = revolute_targets[i]

                # NaN protection and limit clipping
                if np.isnan(target):
                    print(f"[Warning] NaN in revolute target {i}, defaulting to 0")
                    target = 0.0
                if lo < hi and not np.isinf([lo, hi]).any():
                    target = np.clip(target, lo, hi)

                kwargs = {
                    "bodyUniqueId": self.id,
                    "jointIndex": idx,
                    "controlMode": p.POSITION_CONTROL,
                    "targetPosition": target,
                    "force": force,
                    "maxVelocity": vel,
                    "physicsClientId": self.client
                }
                if use_gains:
                    kwargs["positionGain"] = 0.3
                    kwargs["velocityGain"] = 1.0

                p.setJointMotorControl2(**kwargs)

        # === Apply spherical joint controls ===
        if spherical_targets is not None:
            for i, idx in enumerate(self.spherical_joints):
                roll, pitch, yaw = spherical_targets[i]
                if any(np.isnan([roll, pitch, yaw])):
                    print(f"[Warning] NaN in spherical target {i}, defaulting to identity rotation")
                    quat = [0, 0, 0, 1]
                else:
                    quat = p.getQuaternionFromEuler([roll, pitch, yaw])

                kwargs = {
                    "bodyUniqueId": self.id,
                    "jointIndex": idx,
                    "controlMode": p.POSITION_CONTROL,
                    "targetPosition": quat,
                    "force": [500, 500, 500],  # Tunable per joint
                    "physicsClientId": self.client
                }
                if use_gains:
                    kwargs["positionGain"] = 0.3
                    kwargs["velocityGain"] = 1.0

                p.setJointMotorControlMultiDof(**kwargs)



    def get_obs(self):
        # === Base (pelvis) state ===
        pos, orn = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.id, physicsClientId=self.client)
        rotmat = Rotation.from_quat(orn).as_matrix()

        base_lin_vel = np.dot(rotmat.T, lin_vel)     # In robot frame
        base_ang_vel = np.dot(rotmat.T, ang_vel)     # In robot frame
        base_euler = p.getEulerFromQuaternion(orn)   # roll, pitch, yaw #base follows normal rpy convention
        base_z = pos[2]                               # Pelvis height

        # === Joint positions ===
        joint_pos = self.get_joint_position()


        # Parse joint quaternions
        quat_right_hip = joint_pos[0:4]
        angle_right_knee = joint_pos[4] # Revolute joints can be descirbed with a single angle
        quat_left_hip = joint_pos[9:13]
        angle_left_knee = joint_pos[13]
        quat_chest = joint_pos[20:24]

        # Convert selected spherical joints to Euler
        r_hip_roll,_,r_hip_pitch = p.getEulerFromQuaternion(quat_right_hip)
        l_hip_roll,_,l_hip_pitch = p.getEulerFromQuaternion(quat_left_hip)
        chest_roll,_,chest_pitch = p.getEulerFromQuaternion(quat_chest)
        
        # For some reason, spherical joints in sim follow a different order

        # === Assemble observation ===
        obs = np.concatenate([
            base_lin_vel,             # [3]
            base_ang_vel,             # [3]
            [base_euler[0], base_euler[1]],  # roll, pitch [2]
            [base_z],                 # pelvis height [1]
            [chest_roll, chest_pitch], # [2]
            [r_hip_roll, r_hip_pitch, l_hip_roll, l_hip_pitch], # [4]
            [angle_right_knee, angle_left_knee],  # knee joint angles [2]
        ])
        return obs

 
    
    # Meaningless for spherical joints
    def get_joint_limits(self):
        lows, highs = [], []
        for idx in self.joint_index:
            info = p.getJointInfo(self.id, idx, physicsClientId=self.client)
            lows.append(info[8])
            highs.append(info[9])
        return {"low": np.array(lows), "high": np.array(highs)}



    def get_joint_state(self):
        joint_state = p.getJointStatesMultiDof(self.id, self.joint_index, physicsClientId=self.client)

        joint_pos = []
        joint_vel = []
        joint_force = []
        joint_torque = []

        for state in joint_state:
            pos = state[0] if isinstance(state[0], (list, tuple)) else [state[0]]
            vel = state[1] if isinstance(state[1], (list, tuple)) else [state[1]]
            force = state[2] if isinstance(state[2], (list, tuple)) else [state[2]]
            torque = state[3] if isinstance(state[3], (list, tuple)) else [state[3]]

            joint_pos.extend(pos)
            joint_vel.extend(vel)
            joint_force.extend(force)
            joint_torque.extend(torque)

        return (
            np.array(joint_pos),
            np.array(joint_vel),
            np.array(joint_force),
            np.array(joint_torque)
        )

    def get_joint_position(self):
        joint_pos, _, _, _ = self.get_joint_state()
        return joint_pos



    def get_foot_contact(self):
        contacts = []
        for foot in self.feet_index:
            contact_points = p.getContactPoints(bodyA=self.id, bodyB=self.plane, linkIndexA=foot, physicsClientId=self.client)
            normal_force = sum([c[9] for c in contact_points])
            print(f"Foot {foot} contact points: {len(contact_points)}, normal force: {normal_force}")
            contacts.append(1 if normal_force > 150 else 0)
        return np.array(contacts)

    def safety_margin(self):
        pos, orn = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
        z_pos = pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        joint_pos, _, _, _ = self.get_joint_state()

        # Joint limits
        joint_margins = {}
        for i, idx in enumerate(self.joint_index):
            name = p.getJointInfo(self.id, idx, self.client)[1].decode()
            if name in [b"right_elbow", b"left_elbow", b"right_knee", b"left_knee"]:
                # Get joint limits
                lo, hi = p.getJointInfo(self.id, idx, self.client)[8:10]
                val = joint_pos[i]
                joint_margins[f"{name}_min"] = (val - lo) - 0.05  # margin to lower bound
                joint_margins[f"{name}_max"] = (hi - val) - 0.05  # margin to upper bound
            #"abs_roll": 0.7 - abs(roll),

        return {
            "pelvis_height": z_pos - 0.6,
            "abs_pitch": 0.7 - abs(pitch),
            **joint_margins
        }


    def target_margin(self):
        pos, orn = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
        z_pos = pos[2]
        #print("z_pos", z_pos)
        _, pitch, _ = p.getEulerFromQuaternion(orn)
        #print("roll", roll, "pitch", pitch, "yaw", yaw)
        joint_pos, _, _, _ = self.get_joint_state()
        joint_names = [p.getJointInfo(self.id, i)[1].decode() for i in self.joint_index]
        joint_dict = dict(zip(joint_names, joint_pos))

        return {
            "pelvis_height_target": z_pos - 3.5,             # target height is 3.53
            "upright_pitch": 0.2 - abs(pitch),                           # within ±0.2 rad pitch
            "right_knee_neutral": 0.1 - abs(joint_dict["right_knee"]),  # close to 0.0
            "left_knee_neutral": 0.1 - abs(joint_dict["left_knee"]),
            # "right_elbow_neutral": 0.35 - abs(joint_dict["right_elbow"] - (np.pi/2)), # close to pi/2
            # "left_elbow_neutral": 0.35 - abs(joint_dict["left_elbow"] - (np.pi/2)), # close to pi/2
        }

if __name__ == "__main__":
    pass
        
        