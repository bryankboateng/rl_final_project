import pybullet as p
import pybullet_data
import time
import numpy as np

class methods:
    def __init__(self):
        pass

    def get_method(self, method_name):
        if hasattr(self, method_name):
            return getattr(self, method_name)
        else:
            raise ValueError(f"Method {method_name} not found in methods class.")

    def example_method(self):
        return "This is an example method."
    
    def get_controllable_joints(self):
            controllable = []
            for idx in self.joint_index:
                joint_type = p.getJointInfo(self.id, idx, self.client)[2]
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    controllable.append(idx)
            return controllable
        
    def freeze_uncontrolled_joints(self):
        for i in range(p.getNumJoints(self.id, self.client)):
            info = p.getJointInfo(self.id, i, physicsClientId=self.client)
            joint_type = info[2]
            if joint_type == p.JOINT_SPHERICAL:
                # Freeze at current pose
                state = p.getJointStateMultiDof(self.id, i, physicsClientId=self.client)
                pos = state[0] if state[0] else [0, 0, 0, 1]
                vel = state[1] if state[1] else [0, 0, 0]

                p.setJointMotorControlMultiDof(
                    bodyUniqueId=self.id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos,
                    targetVelocity=[0, 0, 0],
                    force=[500, 500, 500],
                    positionGain=1.0,
                    velocityGain=1.0,
                    physicsClientId=self.client
                )


    def apply_position(self, joint_angles):
        self._apply_joint_targets(joint_angles, use_gains=True)

    def _apply_joint_targets(self, targets, use_gains=False):
        if len(targets) != len(self.controllable_joints):
            raise ValueError(f"Target length {len(targets)} does not match controllable joints {len(self.controllable_joints)}")
        
        # Freeze spherical joints
        self.freeze_uncontrolled_joints()
        
        
        for i, idx in enumerate(self.controllable_joints):
            info = p.getJointInfo(self.id, idx, physicsClientId=self.client)


            # if joint_type == p.JOINT_SPHERICAL:
            #     # Handle spherical joints separately
            #     # For now, just skip them
            #     continue
            lower_limit = info[8]
            upper_limit = info[9]
            max_force = info[10]
            max_velocity = info[11]

            # Defensive checks

            assert not np.isnan(targets[i]), f"[ERROR] Target angle NaN at joint index {i}"
            assert not np.isnan(lower_limit), f"[ERROR] NaN lower limit for joint {idx}"
            assert not np.isnan(upper_limit), f"[ERROR] NaN upper limit for joint {idx}"

            # If limits are nonsensical (e.g., -inf, inf), just skip clipping
            if lower_limit > upper_limit or np.isinf([lower_limit, upper_limit]).any():
                clipped_target = targets[i]
            else:
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