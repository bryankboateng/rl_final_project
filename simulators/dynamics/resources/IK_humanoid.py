import numpy as np
import pybullet as p
import time

class IKWalkingController:
    def __init__(self, humanoid, step_height=0.1, step_period=0.8):
        self.humanoid = humanoid
        self.step_height = step_height
        self.step_period = step_period  # full left+right step cycle
        self.phase = 0.0
        self.time = 0.0

        # Save foot indices
        self.left_foot_idx, self.right_foot_idx = humanoid.feet_index

        # Store initial foot positions
        self.left_foot_rest = self.get_foot_world_position(self.left_foot_idx)
        self.right_foot_rest = self.get_foot_world_position(self.right_foot_idx)

    def get_foot_world_position(self, foot_idx):
        link_state = p.getLinkState(self.humanoid.id, foot_idx, computeForwardKinematics=True)
        return np.array(link_state[4])  # world position of foot

    def compute_step_position(self, rest_pos, lift, swing):
        """Generate target foot position for a given phase of the step."""
        target = rest_pos.copy()
        if lift:
            target[2] += self.step_height * np.sin(np.pi * swing)
        return target

    def get_joint_targets(self, dt):
        self.time += dt
        self.phase = (self.time % self.step_period) / self.step_period
        swing_leg = 'left' if self.phase < 0.5 else 'right'
        swing_phase = (self.phase % 0.5) * 2.0  # normalized [0,1]

        # Select target for swing foot
        if swing_leg == 'left':
            foot_idx = self.left_foot_idx
            target = self.compute_step_position(self.left_foot_rest, lift=True, swing=swing_phase)
        else:
            foot_idx = self.right_foot_idx
            target = self.compute_step_position(self.right_foot_rest, lift=True, swing=swing_phase)

        # IK produces a full joint vector — we extract only what's needed
        full_joint_angles = p.calculateInverseKinematics(self.humanoid.id, foot_idx, target)
        return [full_joint_angles[i] for i in self.humanoid.joint_index]

    def step(self, dt=1.0 / 60.0):
        joint_targets = self.get_joint_targets(dt)
        self.humanoid.apply_position(position=joint_targets)
