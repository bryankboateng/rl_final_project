import numpy as np
import pybullet as p
from .base_pybullet_dynamics import BasePybulletDynamics
from typing import Optional, Tuple, Any
from .resources.humanoid import Humanoid
from .resources.force import Force
import time
import matplotlib.pyplot as plt
from jaxlib.xla_extension import DeviceArray
from scipy.spatial.transform import Rotation

class HumanoidDynamicsPybullet(BasePybulletDynamics):

    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        if isinstance(config.obs_dim, dict):
            self.dim_x = config.obs_dim["actor_0"]
        else:
            self.dim_x = config.obs_dim

        self.obsrv_list = config.obsrv_list.ctrl
        self.action_type = config.action_type
        self.action_center = config.action_center

        self.target_list = config.target_margin
        self.safety_list = config.safety_margin

        self.obs_sequence = []
        self.obs_sequence_length = self.obsrv_list.count("obs") if self.obsrv_list and "obs" in self.obsrv_list else 0

        if isinstance(action_space, dict):
            super().__init__(config, action_space["ctrl"])
            self.dim_u = len(action_space["ctrl"])
        else:
            super().__init__(config, action_space)
            self.dim_u = len(action_space)

        self.robot = None
        self.reset()

    def apply_action(self, action):
        if self.action_type == "increment":
            current_angles = np.array(self.robot.get_joint_position())
            target_angles = current_angles + np.array(action)
        elif self.action_type == "center_sampling":
            target_angles = np.array(self.action_center) + np.array(action)
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

        self.robot.apply_action(target_angles)

    def reset(self, **kwargs):
        while True:
            height = kwargs.get("initial_height", 4.0)
            rotation = kwargs.get("initial_rotation", p.getQuaternionFromEuler([1.57, 0, 0]))
            joint_position = kwargs.get("initial_joint_value")
            joint_velocity = kwargs.get("initial_joint_velocity", [0.0] * self.dim_u)
            linear_velocity = kwargs.get("initial_linear_vel", [0.0, 0.0, 0.0])
            angular_velocity = kwargs.get("initial_angular_vel", [0.0, 0.0, 0.0])
            height_mode = kwargs.get("initial_height_reset_type", "stand")
            is_rollout_shielding_reset = kwargs.get("is_rollout_shielding_reset", False)

            if self.robot is None:
                super().reset(**kwargs)
                self.robot = Humanoid(
                    self.client, height=height, orientation=rotation,
                    env_type=None, payload_max=0,
                    target_list=self.target_list, safety_list=self.safety_list, **kwargs
                )
            else:
                p.resetBasePositionAndOrientation(
                    self.robot.id, [0, 0, height], rotation, physicsClientId=self.client
                )

            if joint_position is None:
                joint_position = self.get_random_joint_value()

            if height_mode == "drop":
                self.robot.reset(joint_position)
                self.robot.apply_position(joint_position)
                p.setGravity(0, 0, self.gravity * 0.2, physicsClientId=self.client)
                for _ in range(100):
                    p.stepSimulation(physicsClientId=self.client)
                p.setGravity(0, 0, self.gravity, physicsClientId=self.client)
            elif height_mode == "stand":
                self.robot.reset(np.zeros(self.dim_u))
                traj = np.linspace(self.robot.get_joint_position(), joint_position, 100)
                for pose in traj:
                    self.robot.apply_position(pose)
                    p.stepSimulation(physicsClientId=self.client)

            self.initial_joint_value = joint_position
            self.initial_height = height
            self.initial_rotation = rotation

            p.resetBaseVelocity(
                self.robot.id,
                linearVelocity=linear_velocity,
                angularVelocity=angular_velocity,
                physicsClientId=self.client
            )

            self.initial_linear_vel = linear_velocity
            self.initial_angular_vel = angular_velocity

            base_state = self.robot.get_obs()
            if self.obsrv_list:
                for o in self.obsrv_list:
                    if o == "obs":
                        self.obs_sequence = [list(base_state)] * self.obs_sequence_length
                        base_state = base_state + tuple(sum(self.obs_sequence, []))
                    elif o == "prev_ctrl":
                        base_state = base_state + tuple(joint_position)
            self.state = np.array(base_state, dtype=np.float32)

            if is_rollout_shielding_reset:
                break

            if self.reset_criterion == "failure":
                if min(self.robot.safety_margin().values()) > 0:
                    break
            elif self.reset_criterion == "reach-avoid":
                if min(self.robot.safety_margin().values()) > 0 and min(self.robot.target_margin().values()) < 0:
                    break

    def get_random_joint_value(self):
        if self.action_type == "increment":
            return np.random.uniform(-0.5, 0.5, size=self.dim_u)
        elif self.action_type == "center_sampling":
            return np.array(self.action_center) + np.random.uniform(-0.3, 0.3, size=self.dim_u)
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

    def integrate_forward(
        self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1,
        noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif',
        adversary: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        self.apply_action(control)
        p.stepSimulation(physicsClientId=self.client)

        base_state = self.robot.get_obs()
        if self.obsrv_list:
            for o in self.obsrv_list:
                if o == "obs":
                    assert len(self.obs_sequence) == self.obs_sequence_length
                    base_state = base_state + tuple(sum(self.obs_sequence, []))
                elif o == "prev_ctrl":
                    base_state = base_state + tuple(control)
        self.state = np.array(base_state, dtype=np.float32)

        return self.state, control

    def get_constraints(self):
        return self.robot.safety_margin()

    def get_target_margin(self):
        return self.robot.target_margin()

    def integrate_forward_jax(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return super().integrate_forward_jax(state, control)

    def _integrate_forward(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return super()._integrate_forward(state, control)
