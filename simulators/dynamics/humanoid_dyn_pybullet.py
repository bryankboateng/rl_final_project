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
        self.dim_x = config.obs_dim["actor_0"] if isinstance(config.obs_dim, dict) else config.obs_dim
        self.obsrv_list = getattr(config.obsrv_list, 'ctrl', [])
        #self.obs_sequence_length = self.obsrv_list.count("obs")
        self.obs_sequence = []

        if isinstance(action_space, dict):
            super().__init__(config, action_space["ctrl"])
            self.dim_u = len(action_space["ctrl"])
        else:
            super().__init__(config, action_space)
            self.dim_u = len(action_space)

        self.action_type = config.action_type
        self.action_center = np.array(config.action_center)
        self.target_list = config.target_margin
        self.safety_list = config.safety_margin

        self.robot = None

        self.force_applied_force_vector = None
        self.force_applied_position_vector = None
        self.adv_debug_line_id = None
        self.adversarial_object = None
        self.link_name = getattr(config.force_info, 'link_name', None)
        self.use_arrow_for_adversarial = True

        self.initial_height = None
        self.initial_rotation = None
        self.initial_joint_value = None
        self.initial_joint_vel = None
        self.initial_linear_vel = None
        self.initial_angular_vel = None
        self.initial_action = None

        self.synthetic_symmetrical_dstb = False
        self.reset_count = 0
        self.dstb_array = []

        self.rendered_img = None
        self.state = None
        self.cnt = 0

        self.reset()

    def reset(self, **kwargs):
        self.reset_count = (self.reset_count + 1) % 2 if self.synthetic_symmetrical_dstb else 0
        self.dstb_array = []

        max_attempts = 10
        for attempt in range(max_attempts):
            # --- Required reset criteria ---
            criterion = getattr(self, 'reset_criterion', None)
            is_rollout_shielding_reset = kwargs.get("is_rollout_shielding_reset", False)

            # === Set initial state ===
            height = 4.0
            rotation = p.getQuaternionFromEuler([1.57, 0, 0])

            initial_action = kwargs.get("initial_action", self.get_random_joint_increment())
            self.initial_height = height
            self.initial_rotation = rotation
            self.initial_action = initial_action

            # === Create or reset robot ===
            if self.robot is None:
                super().reset(**kwargs)
                self.robot = Humanoid(self.client, height=height, orientation=rotation,
                                    target_list=self.target_list, safety_list=self.safety_list, **kwargs)
                
            # Always reset to near standing position
            p.resetBasePositionAndOrientation(self.robot.id, [0, 0, height], rotation, physicsClientId=self.client)

            # === Drop Pose ===
            self.robot.apply_position(default=True) #defaults to basic drop pose
            p.setGravity(0, 0, self.gravity * 0.2, physicsClientId=self.client)
            for _ in range(100):
                p.stepSimulation(physicsClientId=self.client)
            p.setGravity(0, 0, self.gravity, physicsClientId=self.client)

            # === Apply initial action ===
            self.robot.apply_action(initial_action)

            # === Build initial observation ===
            base_state = self.robot.get_obs()
            if self.obsrv_list:
                for o in self.obsrv_list:
                    if o == "obs":
                        self.obs_sequence = [list(base_state)] * self.obs_sequence_length
                        base_state = base_state + tuple(sum(self.obs_sequence, []))
                    elif o == "prev_ctrl":
                        base_state = base_state + tuple(initial_action)
            self.state = np.array(base_state, dtype=np.float32)

            if is_rollout_shielding_reset:
                return

            # === Check reset condition ===
            safety = self.robot.safety_margin()
            target = self.robot.target_margin()

            if criterion == "failure" and min(safety.values()) > 0:
                return
            elif criterion == "reach-avoid":
                if min(safety.values()) > 0 and min(target.values()) < 0:
                    return

        raise RuntimeError("[HumanoidDynamicsPybullet] Failed to reset to valid state after multiple attempts.")





    def get_random_joint_value(self):
        if self.action_type == "increment":
            # Safe standing posture within joint limits
            ValueError("Action type 'increment' requires a valid action center.")
        elif self.action_type == "center_sampling":
            return np.array(self.action_center) + self.get_random_joint_increment()
        # apply_joint internally handles clipping
        else:
            raise ValueError(f"Unknown action_type: {self.action_type}")

    def get_random_joint_increment(self):
        return np.array([
            # === Revolute joints ===
            np.random.uniform(-0.5, 0.1),   # right_knee
            np.random.uniform(-0.5, 0.1),   # left_knee
            np.random.uniform(-0.6, 0.6),   # right_elbow
            np.random.uniform(-0.6, 0.6),   # left_elbow

            # === Spherical joints: right_hip (roll, pitch, yaw) ===
            np.random.uniform(-np.pi/3, np.pi/3),
            np.random.uniform(-np.pi/6, np.pi/6),
            np.random.uniform(-np.pi/3, np.pi/3),

            # === right_ankle ===
            np.random.uniform(-np.pi/6, np.pi/6),
            np.random.uniform(-np.pi/6, np.pi/6),
            np.random.uniform(-np.pi/6, np.pi/6),

            # === left_hip ===
            np.random.uniform(-np.pi/3, np.pi/3),
            np.random.uniform(-np.pi/6, np.pi/6),
            np.random.uniform(-np.pi/3, np.pi/3),

            # === left_ankle ===
            np.random.uniform(-np.pi/6, np.pi/6),
            np.random.uniform(-np.pi/6, np.pi/6),
            np.random.uniform(-np.pi/6, np.pi/6),

            # === chest ===
            np.random.uniform(-np.pi/12, np.pi/12),  # roll (limited sway)
            np.random.uniform(-np.pi/6, np.pi/6),    # pitch (moderate lean)
            np.random.uniform(-np.pi/6, np.pi/6),    # yaw (twist)
        ])




    def integrate_forward(
        self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1,
        noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif',
        adversary: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:

        # === Update observation buffer ===
        if self.obs_sequence:
            self.obs_sequence.pop()
            self.obs_sequence.insert(0, list(self.robot.get_obs()))
            assert len(self.obs_sequence) == self.obs_sequence_length

        # === Apply control directly ===
        self.robot.apply_action(control)

        # === Adversarial force application ===
        if adversary is not None and not self.replace_adv_with_dr:
            if self.reset_count % 2 == 1 and self.synthetic_symmetrical_dstb and self.dstb_array:
                adversary = self.dstb_array.pop(0)
                adversary[1] = -adversary[1]  # flip y-axis for symmetry
            self._apply_adversarial_force(adversary[:3], adversary[3:])
        else:
            self._apply_force()

        # === Physics update ===
        p.stepSimulation(physicsClientId=self.client)

        # === Visualization ===
        if self.gui:
            if adversary is not None and not self.replace_adv_with_dr:
                if self.adv_debug_line_id is not None:
                    p.removeUserDebugItem(self.adv_debug_line_id)
                start = self.force_applied_position_vector
                end = start + self.force_applied_force_vector
                link_index = self.robot.get_link_id(self.link_name) if self.link_name else -1
                self.adv_debug_line_id = p.addUserDebugLine(
                    start, end, [0, 0, 1], 2.0, parentObjectUniqueId=self.robot.id,
                    parentLinkIndex=link_index, physicsClientId=self.client
                )
            time.sleep(self.dt)
            if self.video_output_file:
                self._save_frames()
            if hasattr(self, "debugger"):
                self.debugger.cam_and_robotstates(self.robot.id)

        elif self.gui_imaginary:
            self.render()

        # === Rebuild observation ===
        base_state = self.robot.get_obs()
        if self.obsrv_list:
            for o in self.obsrv_list:
                if o == "obs":
                    base_state = base_state + tuple(sum(self.obs_sequence, []))
                elif o == "prev_ctrl":
                    base_state = base_state + tuple(control)

        self.state = np.array(base_state, dtype=np.float32)
        self.cnt += 1

        # === Return depending on adversarial type ===
        if adversary is not None:
            if self.replace_adv_with_dr:
                if self.force != 0:
                    adversary = np.concatenate((self.force_applied_force_vector / self.force, self.force_applied_position_vector))
                else:
                    adversary = np.concatenate((np.zeros(3), self.force_applied_position_vector))
            if self.reset_count % 2 == 0 and self.synthetic_symmetrical_dstb:
                self.dstb_array.append(adversary)
            return self.state, control, adversary
        else:
            return self.state, control


    def render(self):
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((200, 200, 4)))

        # Base information
        robot_id, client_id = self.robot.get_ids()
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.01, farVal=100, physicsClientId=self.client)
        pos, ori = [list(l) for l in p.getBasePositionAndOrientation(robot_id, client_id)]

        pos[0] += 1.0
        pos[1] -= 1.0
        pos[2] += 0.7
        ori = p.getQuaternionFromEuler([0, 0.2, np.pi * 0.8])

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(200, 200, view_matrix, proj_matrix, physicsClientId=self.client)[2]
        frame = np.reshape(frame, (200, 200, 4))
        self.rendered_img.set_data(frame)
        plt.draw()
        plt.axis('off')
        plt.title("Rollout imagine env")
        plt.pause(.00001)

    def get_constraints(self):
        return self.robot.safety_margin()

    def get_target_margin(self):
        return self.robot.target_margin()

    def integrate_forward_jax(self, state: DeviceArray, control: DeviceArray) -> Tuple[DeviceArray, DeviceArray]:
        return super().integrate_forward_jax(state, control)

    def _integrate_forward(self, state: DeviceArray, control: DeviceArray) -> DeviceArray:
        return super()._integrate_forward(state, control)
