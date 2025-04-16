# --------------------------------------------------------
# ISAACS : Iterative Soft Adversarial Actor-Critic for Safety
# --------------------------------------------------------

from typing import List, Dict, Union, Optional, Tuple
import torch
import copy
import numpy as np

from base_training import BaseTraining
from utils import Batch
from simulators import BaseEnv
from simulators.vec_env.vec_env import VecEnvBase
from simulators.policy import RandomPolicy

class ISAACS(BaseTraining):
    """
    ISAACS agent with interleaved training and rollout.

    This class trains a controller (ctrl) and a disturbance agent (dstb) jointly in a
    zero-sum setting using Soft Actor-Critic (SAC), with alternating updates and a
    performance-based leaderboard to maintain best checkpoints.
    """

    def __init__(self, cfg_solver, cfg_arch, seed: int):
        """
        Initializes all components, including agents, networks, leaderboard, and metrics.

        Args:

            cfg_solver: Solver parameters (learning rates, timing, replay buffer)
            cfg_arch: Network definitions for actor/critic
            seed: Random seed
        """
        super().__init__(cfg_solver, cfg_arch, seed)
        # TODO: Set up control and disturbance agents from self.actors dict
        self.cfg_solver = cfg_solver
        self.cfg_arch = cfg_arch
        self.seed = seed
        
        # self.ctrl = ...
        self.ctrl = self.actors['ctrl']
        # self.dstb = ...
        self.dstb = self.actors['dstb']
        # self.critic = ...
        self.critic = self.actors['critic']


        # Checkpoint lists (step numbers)
        self.ctrl_ckpts = []
        self.dstb_ckpts = []

        # Initialize fixed policies
        # self.rnd_ctrl_policy = ...
        self.rnd_ctrl_policy = RandomPolicy(id='rnd_ctrl', action_range=self.ctrl.action_range, seed=self.seed)
        # self.rnd_dstb_policy = ...
        self.rnd_dstb_policy = RandomPolicy(id='rnd_dstb', action_range=self.dstb.action_range, seed=self.seed)
        # self.dummy_dstb_policy = ...
        self.dummy_dstb_policy = lambda obs: np.zeros(self.dstb.action_dim)
        # Evaluation policy copies (for checkpoint loading)
        # self.ctrl_eval = ...
        self.ctrl_eval = copy.deepcopy(self.ctrl)
        # self.dstb_eval = ...
        self.dstb_eval = copy.deepcopy(self.dstb)
    

        # Disturbance sampling distribution settings
        self.softmax_rationality = cfg_solver.softmax_rationality
        self.ctrl_update_ratio = cfg_solver.ctrl_update_ratio
        self.cnt_dstb_updates = 0  # Counts how many dstb updates since last ctrl update

        # Leaderboard: shape (K_ctrl + 1, K_dstb + 2, metrics)
        self.leaderboard = -np.inf * np.ones((self.cfg_solver.K_ctrl + 1, self.cfg_solver.K_dstb + 2, 3))

    def sample(self, obsrv_all: torch.Tensor) -> List[Dict[str, np.ndarray]]:
        """
        Samples control and disturbance actions given current observations.

        Uses:
        - random policies if within warmup phase
        - learned policies otherwise
        - each environment may use a different dstb sampler

        Returns:
            List of action dictionaries with 'ctrl' and 'dstb' keys
        """
        # Determine whether to use random policies (warmup) or learned policies.
        # if self.cnt_step < self.cfg_solver.warmup_steps:
        #     # Use random policies for both ctrl and dstb
        #     action_all = [
        #         {'ctrl': self.rnd_ctrl_policy(obsrv), 'dstb': self.rnd_dstb_policy(obsrv)}
        #         for obsrv in obsrv_all
        #     ]
        # else:
        #     # Use learned policies for ctrl and dstb
        #     action_all = [
        #         {'ctrl': self.ctrl(obsrv), 'dstb': self.dstb(obsrv)}
        #         for obsrv in obsrv_all
        #     ]

        # Determine whether to use random policies (warmup) or learned policies.
        if self.cnt_step < self.cfg_solver.warmup_steps:
            # Warmup: use fixed random policies.
            ctrl_actions = self.rnd_ctrl_policy(obsrv_all)
            dstb_actions = self.rnd_dstb_policy(obsrv_all)
        else:
            # After warmup: use learned policies.
            ctrl_actions = self.ctrl(obsrv_all)
            dstb_actions = self.dstb(obsrv_all)
            ctrl_actions = ctrl_actions.detach().cpu().numpy()
            dstb_actions = dstb_actions.detach().cpu().numpy()

        # Create a list of dictionaries, one per environment.
        # Each dictionary has keys 'ctrl' and 'dstb'.
        action_all = []
        batch_size = obsrv_all.shape[0]
        for idx in range(batch_size):
            action_all.append({
                'ctrl': ctrl_actions[idx],
                'dstb': dstb_actions[idx]
            })
        return action_all

    def interact(
        self,
        rollout_env: Union[BaseEnv, VecEnvBase],
        obsrv_all: torch.Tensor,
        action_all: List[Dict[str, np.ndarray]],
    ) -> torch.Tensor:
        """
        Interacts with environment using sampled actions.

        Stores transitions in replay buffer, tracks safety violations, resets environments that are done,
        and resamples disturbance agent for next episode.

        Returns:
            obsrv_nxt_all: Tensor of next observations
        """
        # TODO: Step environment, store to replay buffer, update counters
        # pass
        obsrv_nxt_all, rewards, dones, infos = rollout_env.step(action_all)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, dtype=torch.bool)
        # safety_violations = np.array([info.get('safety_violation', 0.0) for info in infos])

        # Create a Batch instance with the transition.
        transition = Batch(
            obs=obsrv_all,
            actions=action_all,
            rewards=rewards_tensor,
            next_obs=obsrv_nxt_all,
            dones=dones_tensor,
            infos=infos,
        )

        # Store transition in replay buffer
        self.memory.update(transition)
        # Update global step counter.
        self.cnt_step += 1

        # If any environments are done, reset them.
        if any(dones):
            # Identify which indices (environments) are done.
            done_indices = [i for i, done in enumerate(dones) if done]
            # Reset the done environments.
            obsrv_reset = rollout_env.reset(indices=done_indices)
            # Replace the corresponding entries in the next observations with the reset observations.
            # This ensures that the replay buffer and the next state used by the agent reflect the reset state.
            for idx, reset_obs in zip(done_indices, obsrv_reset):
                obsrv_nxt_all[idx] = reset_obs

            self.dstb.reset(done_indices)
            # self.dstb_sampler_list[done_indices] = self.get_dstb_sampler()

        return obsrv_nxt_all

    def update(self):
        """
        Runs training updates based on buffer data.

        Updates are triggered when:
        - self.cnt_step >= self.min_steps_b4_opt
        - self.cnt_opt_period >= self.opt_period

        Each update cycle:
        - Runs `self.num_updates_per_opt` gradient steps
        - Updates dstb every step
        - Updates ctrl only every `ctrl_update_ratio` steps

        Logs loss metrics and updates target networks.
        """
        # TODO: Sample batch and call update_one()
        # Check `cnt_opt_period`, reset as needed
        # pass
        # Check if conditions are met for starting updates.
        if self.cnt_step < self.cfg_solver.min_steps_b4_opt or self.cnt_opt_period < self.cfg_solver.opt_period:
            return
        
        # Accumulate loss metrics for logging.
        accumulated_losses = {
            "q_loss": 0.0,
            "ctrl_loss": 0.0,
            "ent_ctrl_loss": 0.0,
            "alpha_ctrl_loss": 0.0,
            "dstb_loss": 0.0,
            "ent_dstb_loss": 0.0,
            "alpha_dstb_loss": 0.0
        }
        ctrl_updates = 0

        # Number of update iterations to perform in this cycle.
        num_updates = self.cfg_solver.num_updates_per_opt

        for update_idx in range(num_updates):
            # Determine whether to update ctrl on this step.
            update_ctrl = (update_idx % self.cfg_solver.ctrl_update_ratio == 0)
            # Always update disturbance agent.
            update_dstb = True  

            # Sample a batch from the replay buffer.
            batch = self.memory.sample(self.cfg_solver.batch_size)

            # Perform a single update step and obtain loss values.
            # The update_one method should return a tuple with losses:
            # (q_loss, ctrl_loss, ent_ctrl_loss, alpha_ctrl_loss, dstb_loss, ent_dstb_loss, alpha_dstb_loss)
            losses = self.update_one(batch, update_idx, update_ctrl, update_dstb)

            # Accumulate losses for logging.
            if update_ctrl:
                accumulated_losses["ctrl_loss"] += losses[1]
                accumulated_losses["ent_ctrl_loss"] += losses[2]
                accumulated_losses["alpha_ctrl_loss"] += losses[3]
                ctrl_updates += 1
            accumulated_losses["q_loss"] += losses[0]
            accumulated_losses["dstb_loss"] += losses[4]
            accumulated_losses["ent_dstb_loss"] += losses[5]
            accumulated_losses["alpha_dstb_loss"] += losses[6]

            # Compute average losses for logging.
            avg_ctrl_loss = accumulated_losses["ctrl_loss"] / ctrl_updates if ctrl_updates > 0 else 0
            avg_q_loss = accumulated_losses["q_loss"] / num_updates
            avg_dstb_loss = accumulated_losses["dstb_loss"] / num_updates

        # Where should we log the avg losses?

        # After an update cycle, reset the optimization period counter.
        self.cnt_opt_period = 0

# I dont think the current critic network and target critic network are properly implemented right now.

    def update_one(self, batch: Batch, timer: int, update_ctrl: bool, update_dstb: bool) -> Tuple[float, ...]:
        """
        Performs one update step for critic and optionally ctrl and dstb.

        Uses SAC-style updates with entropy regularization and soft target updates.

        Args:
            batch: Transition batch from replay buffer
            timer: Update step index within the current optimization cycle
            update_ctrl: Flag to indicate ctrl should be updated this step
            update_dstb: Flag to update dstb

        Returns:
            Tuple of loss metrics: (q, ctrl, ent_ctrl, alpha_ctrl, dstb, ent_dstb, alpha_dstb)
        """
        # TODO: Compute gradients and losses for each network component
        # pass

        # Unpack the batch.
        obs = batch.obs  # shape: (B, obs_dim)

        # Extract control and disturbance actions.
        ctrl_actions = torch.stack([torch.tensor(a['ctrl'], dtype=torch.float32) for a in batch.actions])
        dstb_actions = torch.stack([torch.tensor(a['dstb'], dtype=torch.float32) for a in batch.actions])
        # Concatenate along the last dimension for the critic.
        current_actions = torch.cat([ctrl_actions, dstb_actions], dim=-1)
        # Extract rewards, next observations, and done flags.
        rewards = batch.rewards  # shape: (B,)
        next_obs = batch.next_obs  # shape: (B, obs_dim)
        dones = batch.dones.float()  # shape: (B,)

        gamma = self.cfg_solver.gamma  # discount factor

        # ----- Critic Update -----
        # Sample next actions and corresponding log probabilities from both policies.
        next_ctrl_actions, next_ctrl_log_prob = self.ctrl.sample(next_obs)
        next_dstb_actions, _ = self.dstb.sample(next_obs)
        next_actions = torch.cat([next_ctrl_actions, next_dstb_actions], dim=-1)

        # Compute target Q-values using target networks
        q1_next, q2_next = self.critic.target(next_obs, next_actions)
        min_q_next = torch.min(q1_next, q2_next)

        # Use control's entropy bonus in the target.
        alpha_ctrl = self.ctrl.alpha  # typically computed as exp(log_alpha)
        target_q = rewards + gamma * (1 - dones) * (min_q_next - alpha_ctrl * next_ctrl_log_prob)
        target_q = target_q.detach()

        # Compute current Q estimates.
        q1, q2 = self.critic(obs, current_actions)
        critic_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(q2, target_q)

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # ----- Control Policy Update -----
        if update_ctrl:
            # Sample new control actions for current observations along with log probabilities.
            new_ctrl_actions, new_ctrl_log_prob = self.ctrl.sample(obs)
            # For the disturbance, either use the current policy or a fixed sample.
            new_dstb_actions, _ = self.dstb.sample(obs)
            new_actions = torch.cat([new_ctrl_actions, new_dstb_actions], dim=-1)

            # Compute Q-values for the current control actions.
            q1_ctrl, q2_ctrl = self.critic(obs, new_actions)
            min_q_ctrl = torch.min(q1_ctrl, q2_ctrl)

            # SAC actor loss for ctrl: aiming to maximize Q + entropy.
            ctrl_loss = (self.ctrl.alpha * new_ctrl_log_prob - min_q_ctrl).mean()

            # Update control policy.
            self.ctrl.optimizer.zero_grad()
            ctrl_loss.backward()
            self.ctrl.optimizer.step()

            # Automatic entropy tuning for ctrl.
            target_entropy_ctrl = self.cfg_solver.target_entropy
            log_alpha_ctrl = self.ctrl.log_alpha  # assume this is a learnable parameter
            alpha_ctrl_loss = (-log_alpha_ctrl * (new_ctrl_log_prob + target_entropy_ctrl).detach()).mean()

            self.ctrl.alpha_optimizer.zero_grad()
            alpha_ctrl_loss.backward()
            self.ctrl.alpha_optimizer.step()
        else:
            ctrl_loss = torch.tensor(0.0, device=obs.device)
            new_ctrl_log_prob = torch.tensor(0.0, device=obs.device)
            alpha_ctrl_loss = torch.tensor(0.0, device=obs.device)

        # ----- Disturbance Policy Update -----
        if update_dstb:
            # Sample new disturbance actions for current observations.
            new_dstb_actions, new_dstb_log_prob = self.dstb.sample(obs)
            # For stability, use a fixed control action (sampled with no gradient flow).
            with torch.no_grad():
                fixed_ctrl_actions, _ = self.ctrl.sample(obs)
            new_actions_dstb = torch.cat([fixed_ctrl_actions, new_dstb_actions], dim=-1)

            # Compute Q-values for the current disturbance actions.
            q1_dstb, q2_dstb = self.critic(obs, new_actions_dstb)
            min_q_dstb = torch.min(q1_dstb, q2_dstb)

            # For an adversarial disturbance agent, we reverse the sign of the typical SAC objective.
            # The disturbance policy seeks to increase the critic's Q-value (i.e. worsen safety).
            dstb_loss = (-min_q_dstb - self.dstb.alpha * new_dstb_log_prob).mean()

            # Update disturbance policy.
            self.dstb.optimizer.zero_grad()
            dstb_loss.backward()
            self.dstb.optimizer.step()

            # Automatic entropy tuning for disturbance.
            target_entropy_dstb = self.cfg_solver.target_entropy_dstb  # defined in config
            log_alpha_dstb = self.dstb.log_alpha  # learnable parameter
            alpha_dstb_loss = (log_alpha_dstb * (new_dstb_log_prob - target_entropy_dstb).detach()).mean()
            self.dstb.alpha_optimizer.zero_grad()
            alpha_dstb_loss.backward()
            self.dstb.alpha_optimizer.step()
        else:
            dstb_loss = torch.tensor(0.0, device=obs.device)
            new_dstb_log_prob = torch.tensor(0.0, device=obs.device)
            alpha_dstb_loss = torch.tensor(0.0, device=obs.device)

        # ----------------- Soft Update of Target Networks ----------------- #
        self.critic.update_target()

        # Return a tuple of loss values as floats.
        return (
            critic_loss.item(),        # q_loss
            ctrl_loss.item(),          # ctrl_loss
            new_ctrl_log_prob.item(),  # ent_ctrl (proxy via log_prob)
            alpha_ctrl_loss.item(),    # alpha_ctrl_loss
            dstb_loss.item(),          # dstb_loss
            new_dstb_log_prob.item(),  # ent_dstb (proxy via log_prob)
            alpha_dstb_loss.item()     # alpha_dstb_loss
        )


    def eval(self, env: BaseEnv, rollout_env: Union[BaseEnv, VecEnvBase], eval_callback, init_eval: bool = False) -> bool:
        """
        Evaluates the current policy against saved checkpoints and dummy adversary.

        Evaluation is triggered:
        - If init_eval is True (before training begins)
        - If cnt_eval_period >= eval_period

        Updates leaderboard metrics and logs scores to wandb.

        Returns:
            True if evaluation occurred, False otherwise
        """
        # TODO: Loop over ctrl/dstb ckpts and evaluate matchups
        # pass
        # Check if evaluation conditions are met.
        if not init_eval and self.cnt_eval_period < self.cfg_solver.eval_period:
            return False
        
        # NOT IMPLEMENTED YET
        # Number of episodes to run per evaluation matchup.
        num_episodes = self.cfg_solver.eval_episodes

        # Dictionary to store evaluation scores.
        eval_scores = {}

        # --- Evaluate matchups between stored checkpoint policies ---
        for ctrl_idx, ctrl_ckpt in enumerate(self.ctrl_ckpts):
            # Restore the control checkpoint into the evaluation copy.
            self.ctrl_eval.restore(ctrl_ckpt, self.cfg_solver.model_folder)

            for dstb_idx, dstb_ckpt in enumerate(self.dstb_ckpts):
                self.dstb_eval.restore(dstb_ckpt, self.cfg_solver.model_folder)
                matchup_score = self.evaluate_matchup(self.ctrl_eval, self.dstb_eval, env, num_episodes)
                # Use indices in the key names for clarity.
                key = f"ctrl_idx_{ctrl_idx}_vs_dstb_idx_{dstb_idx}"
                eval_scores[key] = matchup_score

        ADD TO LEADERBOARD!!!!

        # Reset evaluation period counter.
        self.cnt_eval_period = 0

        return True


    def evaluate_matchup(self, ctrl_policy, dstb_policy, env: BaseEnv, num_episodes: int) -> float:
        """
        Evaluates a matchup between a control policy and a disturbance policy over a fixed number of episodes.
        
        Args:
            ctrl_policy: A policy (or an object with a get_action or sample method) for controlling the agent.
            dstb_policy: A policy (or lambda/dummy function) for generating disturbance actions.
            env: The environment to run the evaluation on.
            num_episodes: Number of episodes for the evaluation.

        Returns:
            A scalar evaluation score (e.g., average win rate or safety metric).
        """
        total_score = 0.0
        for _ in range(num_episodes):
            obs = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                # Obtain control action (assume ctrl_policy returns an action) and disturbance action.
                ctrl_action = ctrl_policy.get_action(obs)[0] 
                dstb_action = dstb_policy.get_action(obs)[0]
                # Combine actions into a single dictionary.
                action = {"ctrl": ctrl_action, "dstb": dstb_action}
                obs, reward, done, info = env.step(action)
                episode_score += reward
            total_score += episode_score
        average_score = total_score / num_episodes
        return average_score

    


    def update_hyper_param(self):
        """
        Updates learning rate, discount factor, or entropy alpha values dynamically.

        Typically called every timestep.
        """
        # TODO: Update hyperparameters using ctrl, dstb, and critic methods
        pass

    def prune_leaderboard(self):
        """
        Maintains top-k models in leaderboard by pruning worst performers.

        Uses average win rate to evaluate whether current checkpoint is worth keeping.
        """
        # TODO: Update self.ctrl_ckpts and self.dstb_ckpts, and call .save()/.remove()
        pass

    def save(self, max_model: Optional[int] = None):
        """
        Saves ctrl, dstb, and critic models to disk under model_folder.
        """
        # TODO: Save networks with current self.cnt_step
        pass

    def value(self, obsrv: np.ndarray, append: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes value of state using current ctrl + dstb + critic.

        Used during evaluation callback.

        Returns:
            Value estimates as a NumPy array
        """
        # TODO: Forward pass through critic using composed actions
        pass


