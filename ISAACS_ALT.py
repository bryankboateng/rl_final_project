# --------------------------------------------------------
# ISAACS (Assignment Version): Iterative Soft Adversarial Actor-Critic for Safety
# --------------------------------------------------------

from typing import List, Dict, Union, Optional, Tuple
import torch
import numpy as np

# Dear Bryan: I added these 6 imports. I needed them for how I implemented the code but I could be wrong.
import copy
from simulators.policy import RandomPolicy
from actors_and_critics import Actor
from typing import Callable
import wandb
import os
#-----------------------------------------------------------------
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
        self.cfg_solver = cfg_solver
        self.cfg_arch = cfg_arch
        self.seed = seed

        self.aux_metric = cfg_solver.eval.aux_metric
        self.save_top_k_ctrl = int(cfg_solver.save_top_k.ctrl)
        self.save_top_k_dstb = int(cfg_solver.save_top_k.dstb)

        # TODO: Create aliases for actor/critic components for ease of access
        # self.ctrl = ...
        self.ctrl = self.actors['ctrl']
        # self.dstb = ...
        self.dstb = self.actors['dstb']
        # self.critic = ...
        self.critic = self.critics['central']

        # TODO: Setup policies for warmup, dummy, eval copies
        # self.rnd_ctrl_policy = ...
        self.rnd_ctrl_policy = RandomPolicy(
            id='rnd_ctrl',
            action_range=np.array(cfg_solver.warmup_action_range.ctrl, dtype=np.float32),
            seed=self.seed)
        # self.dummy_dstb_policy = ...
        self.rnd_dstb_policy = RandomPolicy(
            id='rnd_dstb',
            action_range=np.array(cfg_solver.warmup_action_range.dstb, dtype=np.float32),
            seed=self.seed)
        # self.dummy_dstb_policy = DummyPolicy(id='dummy', action_dim=self.dstb.action_dim)
        self.dummy_dstb_policy = lambda obs: np.zeros(self.dstb.action_dim)
        # self.ctrl_eval = ...
        self.ctrl_eval = copy.deepcopy(self.ctrl)
        # self.dstb_eval = ...
        self.dstb_eval = copy.deepcopy(self.dstb)

        # Checkpoints.
        # Policy checkpoint tracking
        self.ctrl_ckpts = []
        self.dstb_ckpts = []
        # TODO: Initialize leaderboard tensor
        # Shape: (top_k_ctrl + 1, top_k_dstb + 2, 1 + len(metric_list))
        # self.leaderboard = ...
        self.leaderboard = np.full(
            shape=(self.save_top_k_ctrl + 1, 
                   self.save_top_k_dstb + 2, 
                   1 + len(self.aux_metric)), 
            dtype=float,
            fill_value=None)
        
        # Timing control
        self.ctrl_update_ratio = int(cfg_solver.ctrl_update_ratio)
        self.softmax_rationality = float(cfg_solver.softmax_rationality)

        # Dear Bryan: Do we need the dstb_sampler_list? I assume we do but u no mention it
        self.dstb_sampler_list: List[Union[Callable, RandomPolicy,
                                       Actor]] = [self.rnd_dstb_policy for _ in range(self.num_envs)]

    # Dear Bryan: I added this function. I needed it for how I implemented the code but I could be wrong tho.
    def get_dstb_sampler(self) -> Union[Callable, RandomPolicy, Actor]:
        choices = np.append(np.arange(len(self.dstb_ckpts)), -1)  # Dummy dstb.
        logit = np.mean(self.leaderboard[:len(self.ctrl_ckpts), choices, 0], axis=0)

        prob_un = np.exp(-self.softmax_rationality * logit)  # negative here since dstb minimizes.
        prob = prob_un / np.sum(prob_un)
        dstb_ckpt_idx = self.rng.choice(choices, p=prob)

        if dstb_ckpt_idx == -1:
            return self.dummy_dstb_policy
        else:
            dstb_sampler = copy.deepcopy(self.dstb_eval)
            dstb_sampler.restore(self.dstb_ckpts[dstb_ckpt_idx], self.model_folder, verbose=False)
            return dstb_sampler


    def sample(self, 
               obsrv_all: torch.Tensor
               ) -> List[Dict[str, np.ndarray]]:
        """
        Samples actions for control and disturbance agents given observations.

        Args:
            obsrv_all (torch.Tensor): current observaions of all environments.

        If self.cnt_step < self.warmup_steps, random actions are used.

        Returns:
            A list of dictionaries (length = num_envs), each with keys:
              - 'ctrl': np.ndarray of control action
              - 'dstb': np.ndarray of disturbance action
        """
        # TODO: Implement logic to sample ctrl actions and per-env dstb actions
        # pass
        # Get the number of environments
        num_env = obsrv_all.shape[0]
        obsrv_all = obsrv_all.float().to(self.device)

        # Warm‑up: purely random exploration (populate replay buffer)
        if self.cnt_step < self.warmup_steps:
            ctrl_actions, _ = self.rnd_ctrl_policy.get_action(obsrv_all)   # (N, act_dim_ctrl)
            dstb_actions, _ = self.rnd_dstb_policy.get_action(obsrv_all)   # (N, act_dim_dstb)
        else:
            # Regular SAC inference (no grad)
            with torch.no_grad():
                ctrl_actions = self.ctrl.get_action(obsrv_all).cpu().numpy()
                dstb_actions = self.dstb.get_action(obsrv_all).cpu().numpy()

        # Stitch the two action matrices into the list‑of‑dicts format.
        joint_action: list[dict[str, np.ndarray]] = []
        for i in range(num_env):
            joint_action.append({
                'ctrl': ctrl_actions[i],
                'dstb': dstb_actions[i],
            })

        return joint_action


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
        # Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])
        # TODO: Implement environment interaction + buffer update
        # pass
        obsrv_nxt_all, reward_all, done_all, info_all = rollout_env.step(action_all)
        # Convert obsrv_all to numpy arrays for ReplayBuffer compatability.
        obsrv_all = obsrv_all.cpu().numpy()

        for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
            # Create a Batch instance with the transition.
            transition = Batch(
                s=obsrv_all[env_idx],
                a=action_all[env_idx],
                r=reward_all[env_idx],
                s_=obsrv_nxt_all[env_idx],
                done=done_all[env_idx],
                info=info_all[env_idx],
            )

            # Store transition in replay buffer
            self.memory.update(transition)

            # Check if the episode is done and reset the environment.
            if done:
                obsrv_nxt_all[env_idx] = rollout_env.reset_one(index=env_idx)
            
                # Check for safety violations.
                g_x = info['g_x']
                if g_x < 0:
                    self.cnt_safety_violation += 1

                self.cnt_num_episode += 1
                self.dstb_sampler_list[env_idx] = self.get_dstb_sampler()

        # Update records and counters.
        self.violation_record.append(self.cnt_safety_violation)
        self.episode_record.append(self.cnt_num_episode)

        self.cnt_step        += self.num_envs
        self.cnt_opt_period  += self.num_envs
        self.cnt_eval_period += self.num_envs

        # Return the next observations as a torch tensor.
        return torch.as_tensor(obsrv_nxt_all, 
                               device=self.device, 
                               dtype=torch.float32)


    def collect_episodes(self, 
                         rollout_env: Union[BaseEnv, VecEnvBase],
                         num_episodes: int):
        """
        Collects full episodes using current policies before training.

        Args:
            rollout_env: Vectorized environment for parallel rollout
            num_episodes: Number of complete episodes to collect
        """
        # TODO: Reset environments and track per-env episode completions
        # pass
        # Reset Environment
        obsrv_all = rollout_env.reset(seed=self.seed)
        obsrv_all = torch.as_tensor(obsrv_all,
                                    device=self.device,
                                    dtype=torch.float32)
        
        # Episodes finished before this call.
        start_ep_cnt = self.cnt_num_episode

        # Keep rolling until enough new episodes have been collected
        while self.cnt_num_episode - start_ep_cnt < num_episodes:
            joint_action = self.sample(obsrv_all) 
            obsrv_all    = self.interact(rollout_env,
                                          obsrv_all,
                                          joint_action)
        

    def update_one(self, 
                   batch: Batch, 
                   timer: int, 
                   update_ctrl: bool, 
                   update_dstb: bool
                   ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Runs a single update step for critic, ctrl, and/or dstb using a sampled batch.

        Timing parameters:
            - `update_ctrl` is True only if `cnt_dstb_updates % ctrl_update_ratio == 0`
            - `update_dstb` typically always True

        Returns:
            Tuple of losses: (loss_q, loss_ctrl, entropy_ctrl, alpha_ctrl, loss_dstb, entropy_dstb, alpha_dstb)
        """
        # TODO: Implement SAC-style update for each agent
        # pass
        # Unpack batch
        obs              = batch.obsrv
        obs_nxt          = batch.non_final_obsrv_nxt
        non_final_mask   = batch.non_final_mask
        reward           = batch.reward 
        g_x              = batch.info['g_x']
        l_x              = batch.info['l_x']
        binary_cost      = batch.info['binary_cost']
        ctrl_action      = batch.action['ctrl']
        dstb_action      = batch.action['dstb']

        # Critic Update
        self.critic.net.train()
        self.critic.target.train()
        self.ctrl.net.eval()
        self.dstb.net.eval()

        # Gather actions from both agents.
        with torch.no_grad():
            ctrl_action_nxt, _ = self.ctrl.sample(obs_nxt)
            dstb_action_nxt, _ = self.dstb.sample(
                obs_nxt, agents_action={'ctrl': ctrl_action_nxt.cpu().numpy()}
            )
        action      = torch.cat([ctrl_action, dstb_action], dim=-1)
        action_nxt  = torch.cat([ctrl_action_nxt, dstb_action_nxt], dim=-1)

        # Get the critic Q-values and run update.
        q1_crit,  q2_crit  = self.critic.net(obs, action)
        q1_nxt, q2_nxt = self.critic.target(obs_nxt, action_nxt)

        loss_q = self.critic.update(
            q1=q1_crit, 
            q2=q2_crit,
            q1_nxt=q1_nxt, 
            q2_nxt=q2_nxt,
            non_final_mask=non_final_mask,
            reward=reward,
            g_x=g_x, 
            l_x=l_x, 
            binary_cost=binary_cost,
            entropy_motives=0.0,) # zero‑sum, no entropy bonus here
        
        # Set actor alpha update bool.
        update_alpha = self.cnt_step >= self.warmup_steps

        # Set Critic into eval mode.
        self.critic.net.eval()

        # ctrl Actor Update.
        if update_ctrl and timer % self.ctrl_update_ratio == 0:

            # Configure ctrl for training.
            self.ctrl.net.train()
            self.dstb.net.eval()
            self.critic.net.eval()

            # Gather actions from both agents.
            ctrl_act, log_prob_ctrl = self.ctrl.sample(obsrv=obs)
            with torch.no_grad():
                dstb_act = self.dstb.net(obs)
            joint_action = torch.cat([ctrl_act, dstb_act], dim=-1)

            # Get the critic Q-values and run update.
            q1_ctrl, q2_ctrl = self.critic.net(obs, joint_action)
            loss_ctrl, ent_ctrl, alpha_ctrl = self.ctrl.update(
                q1=q1_ctrl, 
                q2=q2_ctrl,
                log_prob=log_prob_ctrl,
                update_alpha=update_alpha)
        else:
            loss_ctrl = ent_ctrl = alpha_ctrl = 0.0

        # dstb Actor Update.
        if update_dstb:

            # Configure dstb for training.
            self.dstb.net.train()
            self.ctrl.net.eval()

            # Gather actions from both agents.
            with torch.no_grad():
                ctrl_act = self.ctrl.net(obs)
            dstb_act, log_prob_dstb = self.dstb.net.sample(obsrv=obs)
            joint_action = torch.cat([ctrl_act, dstb_act], dim=-1)

            # Get the critic Q-values and run update.
            q1_dstb, q2_dstb = self.critic.net(obs, joint_action)
            loss_dstb, ent_dstb, alpha_dstb = self.dstb.update(
                q1=q1_dstb, 
                q2=q2_dstb,
                log_prob=log_prob_dstb,
                update_alpha=update_alpha)
        else:
            loss_dstb = ent_dstb = alpha_dstb = 0.0

        # Update critic target.
        if timer % self.critic.update_target_period == 0:
            self.critic.update_target()
        
        # Restore all networks to eval mode.
        self.ctrl.net.eval()
        self.dstb.net.eval()
        self.critic.net.eval()
        self.critic.target.eval()

        # Return all losses.
        return (float(loss_q), 
                float(loss_ctrl), 
                float(ent_ctrl), 
                float(alpha_ctrl), 
                float(loss_dstb), 
                float(ent_dstb), 
                float(alpha_dstb))
        

    def update(self):
        """
        Performs a full round of optimization steps after episode collection.

        - Runs `self.num_updates_per_opt` update steps
        - Each step may update ctrl (every `ctrl_update_ratio`) and always updates dstb
        - Logs all losses
        """
        # TODO: Loop through update_one(), track running losses
        # pass
        # Check if we need to update the control policy and create logs.
        if self.cnt_dstb_updates == self.ctrl_update_ratio:
            update_ctrl = True
            self.cnt_dstb_updates = 0
            loss_ctrl_list          = []
            loss_ent_ctrl_list      = []
            loss_alpha_ctrl_list    = []
        else:
            update_ctrl = False

        loss_q_list             = []
        loss_dstb_list          = []
        loss_ent_dstb_list      = []
        loss_alpha_dstb_list    = []

        # reset timer
        self.cnt_opt_period = 0

        for timer in range(self.num_updates_per_opt):
            invalid_batch = True
            attempt = 0
            # Sample a batch from the replay buffer and check validity.
            while invalid_batch and attempt < 10:
                batch = self.memory.sample(self.batch_size)
                invalid_batch = torch.logical_not(torch.any(batch.non_final_mask))
                attempt += 1
            if invalid_batch:
                print("Invalid batch sampled, skipping update.")
                continue

            # Call update_one() to perform a single update step.
            loss_q, loss_ctrl, ent_ctrl, alpha_ctrl, loss_dstb, ent_dstb, alpha_dstb = self.update_one(
                batch=batch,
                timer=timer,
                update_ctrl=update_ctrl,
                update_dstb=True)
            
            # Append losses to lists.
            loss_q_list.append(loss_q)

            if update_ctrl and timer % self.ctrl.update_period == 0:
                loss_ctrl_list.append(loss_ctrl)
                loss_ent_ctrl_list.append(ent_ctrl)
                loss_alpha_ctrl_list.append(alpha_ctrl)

            if timer % self.dstb.update_period == 0:
                loss_dstb_list.append(loss_dstb)
                loss_ent_dstb_list.append(ent_dstb)
                loss_alpha_dstb_list.append(alpha_dstb)

            loss_q_mean = np.mean(loss_q_list)
            loss_dstb_mean = np.mean(loss_dstb_list)
            loss_ent_dstb_mean = np.mean(loss_ent_dstb_list)
            loss_alpha_dstb_mean = np.mean(loss_alpha_dstb_list)

            if update_ctrl:
                loss_ctrl_mean = np.mean(loss_ctrl_list)
                loss_ent_ctrl_mean = np.mean(loss_ent_ctrl_list)
                loss_alpha_ctrl_mean = np.mean(loss_alpha_ctrl_list)
            else:
                loss_ctrl_mean = loss_ent_ctrl_mean = loss_alpha_ctrl_mean = None

            # Log the losses into wandb.
            log_dict = {
                "loss/critic":                  loss_q_mean,
                "loss/dstb":                    loss_dstb_mean,
                "loss/entropy_dstb":            loss_ent_dstb_mean,
                "loss/alpha_dstb":              loss_alpha_dstb_mean,
                "metrics/cnt_safety_violation": self.cnt_safety_violation,
                "metrics/cnt_num_episode":      self.cnt_num_episode,
                "hyper_parameters/gamma":       self.critic.gamma,
                "hyper_parameters/alpha_dstb":  self.dstb.alpha,
                "hyper_parameters/alpha_ctrl":  self.ctrl.alpha}
            if update_ctrl:
                log_dict.update({
                    "loss/ctrl":            loss_ctrl_mean,
                    "loss/entropy_ctrl":    loss_ent_ctrl_mean,
                    "loss/alpha_ctrl":      loss_alpha_ctrl_mean})
            wandb.log(log_dict, step=self.cnt_step, commit=False)

        self.cnt_dstb_updates += 1


    def eval(self, 
             env, 
             rollout_env, 
             eval_callback, 
             init_eval: bool = False) -> bool:
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
        # pass
        # Check if we need to evaluate the policies.
        if (self.cnt_eval_period < self.eval_period) and not init_eval:
            return False
        
        # Reset counter.
        self.cnt_eval_period = 0

        # Get index of the current ctrl/dstb policies.
        current_ctrl_idx = len(self.ctrl_ckpts)
        current_dstb_idx = len(self.dstb_ckpts)

        # Current dstb vs. each saved ctrl‑ckpt
        self.dstb_eval.update_policy(self.dstb)
        for ctrl_ckpt_idx, ctrl_ckpt in enumerate(self.ctrl_ckpts):
            
            # Dear Bryan: I added this code from the original ISAACS. IDK how else to accomplish this and am not sure if this is correct for our case.
            self.ctrl_eval.restore(ctrl_ckpt, self.model_folder, verbose=False)
            env_policy: Actor = env.agent.policy
            env_policy.update_policy(self.ctrl_eval)
            if self.num_envs > 1:
                for agent in self.agent_copy_list:
                    agent_policy: Actor = agent.policy
                    agent_policy.update_policy(self.ctrl_eval)
                rollout_env.set_attr("agent", self.agent_copy_list, value_batch=True)
            else:
                rollout_env_policy: Actor = rollout_env.agent.policy
                rollout_env_policy.update_policy(self.ctrl_eval)
            
            eval_results = eval_callback(env = env, 
                                         rollout_env = rollout_env, 
                                         value_fn=self.value,
                                         adversary=self.dstb_eval,
                                         fig_path=os.path.join(self.figure_folder, f"{ctrl_ckpt}_{self.cnt_step}.png"))
            #---------------------------------------------------------------------------
            self.leaderboard[ctrl_ckpt_idx, current_dstb_idx, 0] = eval_results[self.eval_metric]
            for metric_idx, aux_metric_name in enumerate(self.aux_metric):
                self.leaderboard[ctrl_ckpt_idx, current_dstb_idx, 1 + metric_idx] = eval_results[aux_metric_name]

        # Current ctrl vs. each saved dstb‑ckpt
        self.ctrl_eval.update_policy(self.ctrl)
        env_policy: Actor = env.agent.policy
        env_policy.update_policy(self.ctrl_eval)
        if self.num_envs > 1:
            for agent in self.agent_copy_list:
                agent_policy: Actor = agent.policy
                agent_policy.update_policy(self.ctrl_eval)
            rollout_env.set_attr("agent", self.agent_copy_list, value_batch=True)
        else:
            rollout_env_policy: Actor = rollout_env.agent.policy
            rollout_env_policy.update_policy(self.ctrl_eval)

        for dstb_ckpt_idx, dstb_ckpt in enumerate(self.dstb_ckpts):
            self.dstb_eval.restore(dstb_ckpt, self.model_folder, verbose=False)
            eval_results = eval_callback(env = env, 
                                         rollout_env = rollout_env, 
                                         value_fn=self.value,
                                         adversary=self.dstb_eval,
                                         fig_path=os.path.join(self.figure_folder, f"{self.cnt_step}_{dstb_ckpt}.png"))
            self.leaderboard[current_ctrl_idx, dstb_ckpt_idx, 0] = eval_results[self.eval_metric]
            for metric_idx, aux_metric_name in enumerate(self.aux_metric):
                self.leaderboard[current_ctrl_idx, dstb_ckpt_idx, 1 + metric_idx] = eval_results[aux_metric_name]

        # Current ctrl vs. current dstb
        self.dstb_eval.update_policy(self.dstb)
        eval_results = eval_callback(env = env, 
                                    rollout_env = rollout_env, 
                                    value_fn=self.value,
                                    adversary=self.dstb_eval,
                                    fig_path=os.path.join(self.figure_folder, f"{self.cnt_step}_{self.cnt_step}.png"))
        self.leaderboard[current_ctrl_idx, current_dstb_idx, 0] = eval_results[self.eval_metric]
        for metric_idx, aux_metric_name in enumerate(self.aux_metric):
            self.leaderboard[current_ctrl_idx, current_dstb_idx, 1 + metric_idx] = eval_results[aux_metric_name]

        # Current ctrl vs. dummy dstb
        eval_results = eval_callback(
            env=env,
            rollout_env=rollout_env,
            value_fn=self.value,
            adversary=self.dummy_dstb_policy,
            fig_path=os.path.join(self.figure_folder,
                                  f"{self.cnt_step}_dummy.png"))
        self.leaderboard[current_ctrl_idx, -1, 0] = eval_results[self.eval_metric]
        for metric_idx, aux_metric_name in enumerate(self.aux_metric):
            self.leaderboard[current_ctrl_idx, -1, 1 + metric_idx] = eval_results[aux_metric_name]

        # Aggregate Metrics.
        record_log = {}
        record_log[f'eval/{self.eval_metric}_ctrl'] = np.nanmean(self.leaderboard[current_ctrl_idx, :, 0])
        record_log[f'eval/{self.eval_metric}_dstb'] = np.nanmean(self.leaderboard[:, current_dstb_idx, 0])
        for metric_idx, aux_metric_name in enumerate(self.aux_metric):
            record_log[f'eval/{aux_metric_name}_ctrl'] = np.nanmean(self.leaderboard[current_ctrl_idx, :, metric_idx + 1])
            record_log[f'eval/{aux_metric_name}_dstb'] = np.nanmean(self.leaderboard[:, current_dstb_idx, metric_idx + 1])
        self.eval_record.append(list(record_log.values()))
        wandb.log(record_log, step=self.cnt_step, commit=True)

        # Prune the leaderboard.
        self.prune_leaderboard()

        return True


    def prune_leaderboard(self):
        """
        Replaces worst-performing ctrl/dstb policies in leaderboard with current ones.

        This keeps only top-k performing agents based on average win rate.
        """
        # TODO: Replace entries if current agent performs better than worst saved one
        # pass
        # Save the current critic.
        self.critic.save(self.cnt_step, self.model_folder)

        # Remove the worst performing ctrl policy.
        if len(self.ctrl_ckpts) == self.save_top_k_ctrl:
            ctrl_metric = np.nanmean(self.leaderboard[..., 0], axis=1)
            worst_idx = int(np.argmin(ctrl_metric))

            # If the worst isn’t the freshly‑evaluated one, replace it
            if worst_idx != self.save_top_k_ctrl:
                # Delete the worst ctrl policy.
                self.ctrl.remove(self.ctrl_ckpts[worst_idx], self.model_folder)
                # overwrite list entry.
                self.ctrl_ckpts[worst_idx] = self.cnt_step
                # copy leaderboard row (current → slot)
                self.leaderboard[worst_idx] = self.leaderboard[-1]
                # reset spare row to NaNs for the next round
                self.leaderboard[-1] = np.nan
                # dump weights
                self.ctrl.save(self.cnt_step, self.model_folder)
            else:
                # Still room – just append
                self.ctrl_ckpts.append(self.cnt_step)
                self.ctrl.save(self.cnt_step, self.model_folder)

        # Remove the worst performing dstb policy.
        if len(self.dstb_ckpts) == self.save_top_k_dstb:
            # Exclude dummy column when computing averages (:-1)
            dstb_metric = np.nanmean(self.leaderboard[:, :-1, 0], axis=0)
            worst_idx = int(np.argmin(dstb_metric))

            # If the worst isn’t the freshly‑evaluated one, replace it
            if worst_idx != self.save_top_k_dstb:
                # Delete the worst dstb policy.
                self.dstb.remove(self.dstb_ckpts[worst_idx], self.model_folder)
                # overwrite list entry.
                self.dstb_ckpts[worst_idx] = self.cnt_step
                # copy leaderboard row (current → slot)
                self.leaderboard[:, worst_idx] = self.leaderboard[:, -1]
                # reset spare row to NaNs for the next round
                self.leaderboard[:, -1] = np.nan
                # dump weights
                self.dstb.save(self.cnt_step, self.model_folder)
            else:
                # Still room – just append
                self.dstb_ckpts.append(self.cnt_step)
                self.dstb.save(self.cnt_step, self.model_folder)

    def save(self, max_model: Optional[int] = None):
        """
        Saves current ctrl, dstb, and critic to disk.
        """
        # TODO: Call .save() on each network component
        # pass
        self.ctrl.save(self.cnt_step, self.model_folder, max_model)
        self.dstb.save(self.cnt_step, self.model_folder, max_model)
        self.critic.save(self.cnt_step, self.model_folder, max_model)

    def value(self, 
              obsrv: np.ndarray, 
              append: Optional[np.ndarray] = None
              ) -> np.ndarray:
        """
        Estimates value of a state using current ctrl + dstb + critic.

        Used during evaluation to score rollouts.
        """
        # TODO: Forward pass through ctrl, dstb, and critic
        # pass
        obsrv = torch.as_tensor(obsrv, 
                                device=self.device, 
                                dtype=torch.float32)
        with torch.no_grad():
            ctrl_action = self.ctrl.net(obsrv)
            combined_obsrv = torch.cat([obsrv, ctrl_action], dim=-1)
            dstb_action = self.dstb.net(combined_obsrv)
            joint_action = torch.cat([ctrl_action, dstb_action], dim=-1)
            value = self.critic.value(obsrv, joint_action)

        return value.cpu().numpy()
