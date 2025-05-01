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
# 5.         Knee_Joint_State   <       -3.14   - 0.05
# 6.         Knee_Joint_State   >       0      + 0.05
# 7.         Elbow_Joint_State  <       0      - 0.05
# 8.         Elbow_Joint_State  >       3.14   + 0.05

import numpy as np
import pybullet as p
import os
import math
from .utils import *
from scipy.spatial.transform import Rotation


class Humanoid:
    def __init__(self, client, height=4.0, orientation=None, env_type=None, payload_max=0, **kwargs):
        super().__init__(client, height, orientation, env_type, payload_max, **kwargs)

        self.id = p.loadURDF("/Users/bboat/Desktop/rl_final_project/simulators/dynamics/resources/humanoid/humanoid.urdf", [0, 0, height], useFixedBase=False, physicsClientId=self.client)
        self.dim_x = 28  # 3 lin_vel + 3 ang_vel + 10 joint_pos + 10 joint_vel + 2 contacts

        # Make joint lists
        self.joint_index = self.make_joint_list()
        self.feet_index = self.make_feet_list()  # To detect left/right foot contact

        self.last_contact = np.zeros(2, dtype=bool)  # [left_foot_contact, right_foot_contact]
        self.feet_air_time = np.zeros(2, dtype=float)

        self.plane = self.find_plane()

    def find_plane(self):
        for i in range(p.getNumBodies(self.client)):
            name = p.getBodyInfo(i, self.client)[1].decode("ascii")
            if "plane" in name:
                return i
        return -1

    def make_joint_list(self):
        joint_names = [
            b'right_hip', b'right_knee', b'right_ankle',
            b'left_hip', b'left_knee', b'left_ankle',
            b'right_shoulder', b'right_elbow',
            b'left_shoulder', b'left_elbow'
        ]
        joint_list = []
        for jname in joint_names:
            for i in range(p.getNumJoints(self.id, self.client)):
                name = p.getJointInfo(self.id, i, self.client)[1]
                if name == jname:
                    joint_list.append(i)
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

    def reset(self, position, velocity=None):
        for idx in self.joint_index:
            p.resetJointState(self.id, idx, 0.0, 0.0, physicsClientId=self.client)

    def apply_action(self, action):
        """
        Applies an action to the robot joints based on the current control strategy.
        This is typically called during RL rollouts.

        Args:
            action (np.ndarray): Delta or offset for joint angles, depending on action_type.
        """
        if self.action_type == "increment":
            target_angles = np.array(self.get_joint_position()) + np.array(action)
        elif self.action_type == "center_sampling":
            target_angles = np.array(self.center) + np.array(action)
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

        self._apply_joint_targets(target_angles, use_gains=False)


    def apply_position(self, joint_angles):
        """
        Directly sets joint angles with PD gains (used for resets or scripted motion).

        Args:
            joint_angles (np.ndarray): Absolute joint angles.
        """
        self._apply_joint_targets(joint_angles, use_gains=True)


    def _apply_joint_targets(self, targets, use_gains=False):
        """
        Internal method to apply joint targets with optional PD control gains.

        Args:
            targets (np.ndarray): Target joint angles.
            use_gains (bool): Whether to apply explicit PD gains.
        """
        for i, idx in enumerate(self.joint_index):
            info = p.getJointInfo(self.id, idx, physicsClientId=self.client)
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]
            clipped_target = np.clip(targets[i], lower_limit, upper_limit)

            kwargs = {
                "bodyUniqueId": self.id,
                "jointIndex": idx,
                "controlMode": p.POSITION_CONTROL,
                "targetPosition": clipped_target,
                "force": max_force,
                "maxVelocity": max_velocity,
                "physicsClientId": self.client
            }

            if use_gains:
                kwargs["positionGain"] = 0.3
                kwargs["velocityGain"] = 1.0

            p.setJointMotorControl2(**kwargs)


    def get_obs(self):
        pos, ang = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
        lin_vel, ang_vel = p.getBaseVelocity(self.id, physicsClientId=self.client)
        rotmat = Rotation.from_quat(ang).as_matrix()

        robot_body_linear_vel = np.dot(rotmat.T, lin_vel)
        robot_body_angular_vel = np.dot(rotmat.T, ang_vel)

        joint_pos, joint_vel, joint_force, joint_torque = self.get_joint_state()
        contacts = self.get_foot_contact()

        obs = np.concatenate([
            robot_body_linear_vel,
            ang_vel,
            joint_pos,
            joint_vel,
            contacts
        ])
        return obs

    def get_joint_position(self):
        joint_pos, _, _, _ = self.get_joint_state()
        return joint_pos

    def get_joint_state(self):
        joint_state = p.getJointStates(self.id, self.joint_index, physicsClientId=self.client)
        joint_pos = np.array([state[0] for state in joint_state])
        joint_vel = np.array([state[1] for state in joint_state])
        joint_force = np.array([state[2] for state in joint_state])
        joint_torque = np.array([state[3] for state in joint_state])
        return joint_pos, joint_vel, joint_force, joint_torque

    def get_foot_contact(self):
        contacts = []
        for foot in self.feet_index:
            contact_points = p.getContactPoints(bodyA=self.id, bodyB=self.plane, linkIndexA=foot, physicsClientId=self.client)
            normal_force = sum([c[9] for c in contact_points])
            contacts.append(1 if normal_force > 5 else 0)
        return np.array(contacts)

    def safety_margin(self):
        pos, orn = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
        z_pos = pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        joint_pos, _, _, _ = self.get_joint_state()
        joint_names = [p.getJointInfo(self.id, i)[1].decode() for i in self.joint_index]
        joint_dict = dict(zip(joint_names, joint_pos))

        margin = {
            "pelvis_height": z_pos - 0.6,
            "abs_roll": 0.7 - abs(roll),
            "abs_pitch": 0.7 - abs(pitch),
            "right_knee_min": joint_dict["right_knee"] - (-3.14 + 0.05),
            "right_knee_max": 0.0 - joint_dict["right_knee"],
            "left_knee_min": joint_dict["left_knee"] - (-3.14 + 0.05),
            "left_knee_max": 0.0 - joint_dict["left_knee"],
            "right_elbow_min": joint_dict["right_elbow"] - (0.0 + 0.05),
            "right_elbow_max": 3.14 - joint_dict["right_elbow"],
            "left_elbow_min": joint_dict["left_elbow"] - (0.0 + 0.05),
            "left_elbow_max": 3.14 - joint_dict["left_elbow"]
        }
        return margin

    def target_margin(self):
        pos, orn = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
        z_pos = pos[2]
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        joint_pos, _, _, _ = self.get_joint_state()
        joint_names = [p.getJointInfo(self.id, i)[1].decode() for i in self.joint_index]
        joint_dict = dict(zip(joint_names, joint_pos))

        target = {
            "pelvis_height_target": z_pos - 3.4,
            "upright_roll": 0.2 - abs(roll),
            "upright_pitch": 0.2 - abs(pitch),
            "right_knee_neutral": joint_dict["right_knee"] - (-0.2),
            "left_knee_neutral": joint_dict["left_knee"] - (-0.2),
            "right_elbow_neutral": 1.5 - abs(joint_dict["right_elbow"] - 1.5),
            "left_elbow_neutral": 1.5 - abs(joint_dict["left_elbow"] - 1.5)
        }
        return target


