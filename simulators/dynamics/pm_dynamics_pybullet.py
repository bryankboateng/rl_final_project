import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc
from typing import Optional, Tuple, Any
import matplotlib.pyplot as plt
import os
import random
import imageio.v3 as iio



def scale_and_shift(x, old_range, new_range):
    ratio = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    x_new = (x - old_range[0]) * ratio + new_range[0]
    return x_new


class PointMassPyBullet:
    """
    Point Mass Dynamics with PyBullet, including obstacles, lidar sensing, and adversarial interactions.
    """

    def __init__(self, cfg_agent, tmp_action_space, dt=0.1, render=False, num_obstacles=20, goal_pos=(5.0, 5.0), env_bounds=((-6, 6), (-6, 6))):
        self.dt = dt

        
        self.dim_x = 36
        self.state = np.zeros(36)
        
        #self.goal_pos = np.array(goal_pos)

        self.lidar_angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)  # 36 Lidar beams
        self.lidar_range = 5.0  # Max lidar range in meters
        
        self.obs_sequence_len = 5
        
        self.obs_sequence = []
        self.pole = None
        self.bounded = False
        self.d_safe = 2
        self.num_obstacles=num_obstacles
        self.step = 1
        
        # Defines action space of the controller models.
        self.thetadot_range_model = [-1., 1.]
        self.xdot_range_perf_model = [0.2, 1.0]
        self.xdot_range_backup_model = [0.2, 0.5]

        # Defines action space of the real environment.
        self.thetadot_range = [-1., 1.]
        self.xdot_range_perf = [0.2, 1.0]
        self.xdot_range_backup = [0.2, 0.5]
        
        self.client = bc.BulletClient(connection_mode=p.GUI if render else p.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -9.8)
        self.client.setTimeStep(self.dt)
        
        self.env_bounds = env_bounds  # Soft boundary limits
        self.init_world()

    def init_world(self):
        self.obstacles = self.create_obstacles(self.num_obstacles)
        self.robot_id = self.create_agent()
        #self.goal_id = self.create_goal(self.goal_pos)

    def create_agent(self):
        col_shape = self.client.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        vis_shape = self.client.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 0, 1, 1])
        # Reset state make sure obstacles are set in place first
        x, y, theta = self.get_random_valid_position()
        self.dyn_state = np.array([x, y, theta])
        return self.client.createMultiBody(1, col_shape, vis_shape, basePosition=[0, 0, 0])

    def create_obstacles(self, num_obstacles):
        obstacles = []
        x_min, x_max = self.env_bounds[0]
        y_min, y_max = self.env_bounds[1]
        # training bounds was -2,4
        # Expand to check for robustness?
        for _ in range(num_obstacles):
            x, y = np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)
            col_shape = self.client.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2]) #.3 --> .1
            vis_shape = self.client.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2], rgbaColor=[1, 0, 0, 1])
            obs_id = self.client.createMultiBody(0, col_shape, vis_shape, basePosition=[x, y, 0.1])
            obstacles.append((obs_id, [x, y]))
        return obstacles



    def get_control(self, action):
        if action[-1] == 1:
            xdot_range_model = self.xdot_range_perf_model
            xdot_range = self.xdot_range_perf
        else:
            xdot_range_model = self.xdot_range_backup_model
            xdot_range = self.xdot_range_backup

        v = scale_and_shift(action[0], [-1, 1], xdot_range_model)
        v = np.clip(v, xdot_range[0], xdot_range[1])
        w = np.clip(action[1], self.thetadot_range[0], self.thetadot_range[1])
        return v, w


    def delay(self, lidar_obs)->np.ndarray:
        idx = random.randint(0, self.obs_len)
        # assert 0 <= idx <= self.obs_len
        if len(self.obs_sequence) < self.obs_len:
            self.obs_sequence.insert(0, lidar_obs)
            return lidar_obs
       
        self.obs_sequence.pop() # Pop from the back
        self.obs_sequence.insert(0, lidar_obs) # Insert most recent at front
        #assert len(self.obs_sequence) == self.obs_sequence_length
        if self.pole is not None:
            self.pole+=1 # shift by one to account for shift due to popping and inserting
            if idx > self.pole:
                idx = self.pole - 1
        obs = self.obs_sequence[idx]
        self.pole = idx
        return obs
    
    
    def update_obs_sequence(self, lidar_obs: np.ndarray) -> np.ndarray:
        """
        Updates the obs_sequence buffer and returns the stacked observation.

        Returns:
            np.ndarray: Flattened sequence of lidar observations.
        """
        if len(self.obs_sequence) < self.obs_sequence_len:
            self.obs_sequence.insert(0, lidar_obs)
            while len(self.obs_sequence) < self.obs_sequence_len:
                self.obs_sequence.append(lidar_obs)  # pad with same obs
        else:
            self.obs_sequence.pop()
            self.obs_sequence.insert(0, lidar_obs)

        # Return stacked observation (flattened)
        return np.concatenate(self.obs_sequence, axis=0)



    def integrate_forward(
        self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
        noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        v_ctrl, w_ctrl = control
        v_dstb, w_dstb = adversary[:2]
        # Should clip ctrl disturbances and lidar disturbance
        #Coverage issue for monte carlo estimation to solve model-free Q
        v_final = v_ctrl + v_dstb
        w_final = w_ctrl + w_dstb
        action = [v_final, w_final]
        v_final, w_final = self.get_control(action)
    
        x, y, theta = self.dyn_state
        x += v_final * np.cos(theta) * self.dt
        y += v_final * np.sin(theta) * self.dt
        theta = np.mod(theta + w_final * self.dt, 2 * np.pi)
        
        self.dyn_state = np.array([x, y, theta])
        self.client.resetBasePositionAndOrientation(self.robot_id, [x, y, 0], [0, 0, 0, 1])
    
        lidar_noise = adversary[2]
        exteroceptive_state = self.simulate_lidar(lidar_noise)
        self.state = exteroceptive_state
        obs = self.update_obs_sequence(exteroceptive_state) #new addition to facilitate observation sequence for POMPD
        self.obs = obs
        
        #lag_time = int(adversary[-1]) #lag is now fixed with delay
        return obs, control, adversary
            

    
    
    def simulate_lidar(self, use_adv=True, adversary_noise=0.0, hit_threshold=0.2):
        """
        Simulates Lidar readings with a threshold for what is considered a hit.
        
        Args:
            adversary_noise (float): Noise added to Lidar readings.
            hit_threshold (float): If the hit fraction is below this, it is registered as a hit (distance = 0).
            
        Returns:
            np.ndarray: Array of Lidar distances.
        """
        lidar_readings = []
        for angle in self.lidar_angles:
            end_x = self.dyn_state[0] + self.lidar_range * np.cos(angle)
            end_y = self.dyn_state[1] + self.lidar_range * np.sin(angle)
            ray_start = [self.dyn_state[0], self.dyn_state[1], 0.1] #0.1 offset on z-to prevent collision with the robot with ground
            ray_end = [end_x, end_y, 0.1]
            
            hit_results = self.client.rayTest(ray_start, ray_end)
            hit_fraction = hit_results[0][2] # Fraction of the ray length that was hit
            
            # Apply threshold for what is considered a "hit"
            if hit_fraction <= hit_threshold:
                distance = 0.0  # Register as an immediate hit
            else:
                distance = hit_fraction * self.lidar_range
                if use_adv:
                    distance = distance + np.random.normal(0, adversary_noise)

            lidar_readings.append(distance)

        return np.array(lidar_readings)


    def reset(self):
        self.client.resetSimulation()
        self.client.setGravity(0, 0, -9.8)
        self.client.setTimeStep(self.dt)
        self.obstacles = self.create_obstacles(self.num_obstacles)
        self.robot_id = self.create_agent()
        #self.goal_id = self.create_goal(self.goal_pos)
        return self.update_obs_sequence(self.simulate_lidar())



    def boundary_margin(self, state):
        """
        Computes the soft boundary margin.

        Args:
            state (np.ndarray): The agent's state (x, y, theta).

        Returns:
            float: Distance to the closest boundary (≥ 0).
        """
        x, y = state[:2]
        x_min, x_max = self.env_bounds[0]
        y_min, y_max = self.env_bounds[1]

        # Compute distance to each boundary
        dist_to_xmin = x - x_min
        dist_to_xmax = x_max - x
        dist_to_ymin = y - y_min
        dist_to_ymax = y_max - y

        # Minimum distance to any boundary
        boundary_margin = min(dist_to_xmin, dist_to_xmax, dist_to_ymin, dist_to_ymax)
        return max(boundary_margin, 0.0)  # Ensure non-negative

    def get_constraints(self, bounded=False):
        constraints = {}
        true_lidar = self.simulate_lidar(use_adv=False)
        for i, value in enumerate(true_lidar):
            constraints[f"lidar_ray_{i}"] = value
        if bounded:
            constraints["boundary"] = self.boundary_margin(self.dyn_state)
        return constraints
    
    ## Mistakenly tried to use task policy directly in filter training
    # def get_target_margin(self):
    #     return {'goal': self.goal_radius - (np.linalg.norm(self.dyn_state[:2] - self.goal_pos))}
    

    # def get_target_margin(self):
    #     return {'safe_region': -min(self.get_constraints().values())}
    
    def get_target_margin(self):
        return {'safe_region': min(self.get_constraints(bounded=self.bounded).values()) - self.d_safe}



    def get_random_valid_position(self, safe_margin=.5, max_attempts=100):
        """
        Generates a random valid position for the agent within boundaries,
        ensuring it does not overlap with obstacles.

        Args:
            safe_margin (float): Minimum distance from any obstacle.
            max_attempts (int): Maximum attempts before giving up.

        Returns:
            (float, float): Valid (x, y) position for the agent.
        """
        x_min, x_max = self.env_bounds[0]
        y_min, y_max = self.env_bounds[1]

        for _ in range(max_attempts):
            # Sample a random position within boundaries
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)

            # Check if the position is safe (not inside an obstacle)
            safe = True
            for _, (obs_x, obs_y) in self.obstacles:
                if np.linalg.norm([x - obs_x, y - obs_y]) < safe_margin:
                    safe = False
                    break  # If too close to an obstacle, reject and retry

            if safe:
                theta = random.uniform(0, 2*np.pi)
                return x, y, theta  # Return the first valid position found

        raise RuntimeError("Failed to find a valid starting position for the agent after multiple attempts.")

    def capture_snapshot(self, save_every, save_path, done_signal, gif_every=75):
        """
        Captures a 2D top-down snapshot and generates a GIF every `gif_every` steps.

        Args:
            step (int): Current simulation step.
            save_every (int): Interval for saving snapshots.
            save_path (str): Directory to save snapshots.
            gif_every (int): Interval at which GIFs should be generated.
        """
        if self.step % save_every != 0:
            return  # Only save every 'save_every' steps
        x_min, x_max = self.env_bounds[0]
        y_min, y_max = self.env_bounds[1]
        # Ensure directory exists
        os.makedirs(save_path, exist_ok=True)

        self.step += 1

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 6))
        # ax.set_xlim(-2, 6)
        # ax.set_ylim(-2, 6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Step {self.step}")

        # Plot robot (Blue)
        ax.scatter(self.dyn_state[0], self.dyn_state[1], color='blue', s=100, label="Robot")

        # Plot obstacles (Red, single legend entry)
        first_obstacle = True

        for _, (x, y) in self.obstacles:
            

            if first_obstacle:
                ax.scatter(x, y, color='red', s=100, label="Obstacle")
                first_obstacle = False
            else:
                ax.scatter(x, y, color='red', s=100)

        # Plot lidar rays (Black)
        for i, angle in enumerate(self.lidar_angles):
            end_x = self.dyn_state[0] + self.state[i] * np.cos(angle)
            end_y = self.dyn_state[1] + self.state[i] * np.sin(angle)
            ax.plot([self.dyn_state[0], end_x], [self.dyn_state[1], end_y], 'k-', color='gray', alpha=0.5)  

        # Add legend
        ax.legend(loc="upper left")

        # Save snapshot
        snapshot_path = os.path.join(save_path, f"step_{self.step}.png")
        plt.savefig(snapshot_path)
        plt.close(fig)  # Prevent memory leaks
            
        # Generate GIF at specific intervals
        if done_signal:
            gif_index = self.step
            gif_name = f"training_{gif_index}.gif"
            self.create_gif(save_path, gif_name)
            raise KeyboardInterrupt


    def create_gif(self, save_path, gif_name):
        """
        Generates a GIF from saved images.

        Args:
            save_path (str): Directory containing snapshots.
            gif_name (str): Filename for the GIF.
        """
        gif_path = os.path.join(save_path, gif_name)
        image_files = [os.path.join(save_path, f) for f in os.listdir(save_path) if f.endswith(".png")]
        
        def numeric_key(filename):
            # Example: "snapshots/step_6.png" -> "6" -> int("6")
            # Adjust the split logic as needed for your filenames
            base = filename.split("/")[-1]      # "step_6.png"
            number_part = base.split("_")[1]    # "6.png"
            number = number_part.split(".")[0]  # "6"
            return int(number)

        image_files.sort(key=numeric_key)


        if len(image_files) > 1:  # Ensure at least 2 images for a meaningful GIF
            images = [iio.imread(img) for img in image_files]  # Read images into a list
            iio.imwrite(gif_path, images, format="gif", duration=200)  # Save as GIF

            print(f"GIF saved at {gif_path}")


