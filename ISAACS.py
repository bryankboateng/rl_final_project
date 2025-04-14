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

        self.cfg_solver = cfg_solver
        self.cfg_arch = cfg_arch
        self.seed = seed


        # TODO: Set up control and disturbance agents from self.actors dict
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
        self.leaderboard = None  # TODO: define using args.save_top_k.ctrl, .dstb, etc.

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
        # TODO: Use ctrl and dstb policy objects to sample actions
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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


