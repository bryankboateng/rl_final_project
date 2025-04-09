# --------------------------------------------------------
# ISAACS (Assignment Version): Iterative Soft Adversarial Actor-Critic for Safety
# --------------------------------------------------------

from typing import List, Dict, Union, Optional, Tuple
import torch
import numpy as np

from base_training import BaseTraining
from utils import Batch
from simulators import BaseEnv
from simulators.vec_env.vec_env import VecEnvBase


class ISAACS(BaseTraining):
    """
    Skeleton implementation of ISAACS agent (Iterative Soft Adversarial Actor-Critic for Safety).
    
    This agent coordinates the learning between a control agent (ctrl) and an adversarial disturbance agent (dstb),
    updating their policies using SAC and evaluating performance using a tournament-style leaderboard.

    Attributes:
        ctrl: Control policy (actor)
        dstb: Disturbance/adversarial policy (actor)
        critic: Shared critic (Q-function)
        ctrl_ckpts, dstb_ckpts: List of step IDs where top-k models were saved
        leaderboard: A tensor storing evaluation scores between all pairs of policies (ctrl vs dstb)
        softmax_rationality: Float that governs the sharpness of dstb policy sampling distribution
    """
    
    def __init__(self, cfg_solver, cfg_arch, seed: int):
        """
        Initializes ISAACS agent with given config, architecture, and random seed.

        Args:
            cfg_solver: Solver configuration
            cfg_arch: Architecture configuration
            seed: Random seed
        """
        super().__init__(cfg_solver, cfg_arch, seed)


        # TODO: Create aliases for actor/critic components for ease of access
        # self.ctrl = ...
        # self.dstb = ...
        # self.critic = ...

        # TODO: Setup policies for warmup, dummy, eval copies
        # self.rnd_ctrl_policy = ...
        # self.dummy_dstb_policy = ...
        # self.ctrl_eval = ...
        # self.dstb_eval = ...

        # TODO: Initialize leaderboard tensor
        # Shape: (top_k_ctrl + 1, top_k_dstb + 2, 1 + len(metric_list))
        # self.leaderboard = ...

        # Timing control
        self.ctrl_update_ratio = ...
        self.softmax_rationality = ...

        # Policy checkpoint tracking
        self.ctrl_ckpts = []
        self.dstb_ckpts = []

    def sample(self, obsrv_all: torch.Tensor) -> List[Dict[str, np.ndarray]]:
        """
        Samples actions for control and disturbance agents given observations.

        If self.cnt_step < self.warmup_steps, random actions are used.

        Returns:
            A list of dictionaries (length = num_envs), each with keys:
              - 'ctrl': np.ndarray of control action
              - 'dstb': np.ndarray of disturbance action
        """
        # TODO: Implement logic to sample ctrl actions and per-env dstb actions
        pass

    def interact(
        self,
        rollout_env: Union[BaseEnv, VecEnvBase],
        obsrv_all: torch.Tensor,
        action_all: List[Dict[str, np.ndarray]],
    ) -> torch.Tensor:
        """
        Steps the environment forward and stores the resulting transition in replay memory.

        Also tracks episode completions, violations, and updates dstb policy for each finished env.

        Returns:
            Next observations as torch.Tensor (batch)
        """
        # TODO: Implement environment interaction + buffer update
        pass

    def collect_episodes(self, rollout_env, num_episodes: int):
        """
        Collects full episodes using current policies before training.

        Args:
            rollout_env: Vectorized environment for parallel rollout
            num_episodes: Number of complete episodes to collect
        """
        # TODO: Reset environments and track per-env episode completions
        pass

    def update_one(self, batch: Batch, timer: int, update_ctrl: bool, update_dstb: bool) -> Tuple[float, ...]:
        """
        Runs a single update step for critic, ctrl, and/or dstb using a sampled batch.

        Timing parameters:
            - `update_ctrl` is True only if `cnt_dstb_updates % ctrl_update_ratio == 0`
            - `update_dstb` typically always True

        Returns:
            Tuple of losses: (loss_q, loss_ctrl, entropy_ctrl, alpha_ctrl, loss_dstb, entropy_dstb, alpha_dstb)
        """
        # TODO: Implement SAC-style update for each agent
        pass

    def update(self):
        """
        Performs a full round of optimization steps after episode collection.

        - Runs `self.num_updates_per_opt` update steps
        - Each step may update ctrl (every `ctrl_update_ratio`) and always updates dstb
        - Logs all losses
        """
        # TODO: Loop through update_one(), track running losses
        pass

    def eval(self, env, rollout_env, eval_callback, init_eval: bool = False) -> bool:
        """
        Evaluates current ctrl/dstb against stored checkpoints and dummy policies.

        Updates leaderboard with win rates and auxiliary metrics.

        Args:
            eval_callback: Callable that returns eval_results dict from env
            init_eval: Whether this is the first evaluation before training

        Returns:
            True if evaluation occurred, False otherwise
        """
        # TODO: Iterate through all ctrl/dstb ckpts and store results to leaderboard
        pass

    def prune_leaderboard(self):
        """
        Replaces worst-performing ctrl/dstb policies in leaderboard with current ones.

        This keeps only top-k performing agents based on average win rate.
        """
        # TODO: Replace entries if current agent performs better than worst saved one
        pass

    def save(self, max_model: Optional[int] = None):
        """
        Saves current ctrl, dstb, and critic to disk.
        """
        # TODO: Call .save() on each network component
        pass

    def value(self, obsrv: np.ndarray, append: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estimates value of a state using current ctrl + dstb + critic.

        Used during evaluation to score rollouts.
        """
        # TODO: Forward pass through ctrl, dstb, and critic
        pass
