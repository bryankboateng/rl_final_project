# --------------------------------------------------------
# ISAACS : Iterative Soft Adversarial Actor-Critic for Safety
# --------------------------------------------------------

from typing import List, Dict, Union, Optional, Tuple
import torch
import copy
import numpy as np
import wandb
import warnings
import os


from base_training import BaseTraining
from utils import Batch, DummyPolicy
from simulators import BaseZeroSumEnv, BaseEnv
from simulators.vec_env.vec_env import VecEnvBase
from simulators.policy.random_policy import RandomPolicy

class ISAACSTrainer(BaseTraining):
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

        # Set up control and disturbance agents from self.actors dict
        self.ctrl = self.actors['ctrl']
        self.dstb = self.actors['dstb']
        self.critic = self.critics['central']

        self.save_top_k_ctrl = int(cfg_solver.save_top_k.ctrl)
        self.save_top_k_dstb = int(cfg_solver.save_top_k.dstb)

        # Always keeps the dummy dstb (no dstb) and has placeholder for the current
        self.aux_metric = cfg_solver.eval.aux_metric
        self.leaderboard = np.full(
            shape=(self.save_top_k_ctrl + 1, self.save_top_k_dstb + 2, 1 + len(self.aux_metric)), dtype=float,
            fill_value=None
        )

        # Checkpoint lists (step numbers)
        self.ctrl_ckpts = []
        self.dstb_ckpts = []

        # Initialize fixed policies
        self.rnd_ctrl_policy = RandomPolicy(
            id='rnd_ctrl', action_range=np.array(cfg_solver.warmup_action_range.ctrl, dtype=np.float32), seed=seed
        )
        self.dstb_sampler_list = [RandomPolicy(id='rnd_dstb', action_range=np.array(cfg_solver.warmup_action_range.dstb, dtype=np.float32), seed=self.seed)]
        self.dummy_dstb_policy = DummyPolicy(id='dummy_dstb', action_dim=self.dstb.action_dim)
        
        # Evaluation policy copies (for checkpoint loading)
        self.ctrl_eval = copy.deepcopy(self.ctrl)
        self.dstb_eval = copy.deepcopy(self.dstb)
    
        # Disturbance sampling distribution settings
        self.softmax_rationality = cfg_solver.softmax_rationality
        self.ctrl_update_ratio = cfg_solver.ctrl_update_ratio
        self.cnt_dstb_updates = 0 # Counts how many dstb updates since last ctrl update

        # Leaderboard: shape (K_ctrl + 1, K_dstb + 2, metrics)
        self.leaderboard = np.nan * np.ones((self.save_top_k_ctrl + 1, self.save_top_k_dstb + 2, 3))


    def combine_action(self, ctrl_action: torch.Tensor, dstb_action: torch.Tensor) -> torch.Tensor:
        """
        Combines control and disturbance actions into a single tensor.

        Args:
            ctrl_action: Control action tensor.
            dstb_action: Disturbance action tensor.

        Returns:
            Combined action tensor.
        """

        return torch.cat([ctrl_action, dstb_action], dim=-1)


    def get_dstb_sampler(self) -> RandomPolicy:
        """
        Returns a new instance of the disturbance sampler.

        This is used to sample disturbance actions for the current episode.
        """

        dtsb_idxs = np.array(list(range(len(self.dstb_ckpts))) + [-1]) # -1 to account for dummy dstb
        logit = np.mean(self.leaderboard[:len(self.ctrl_ckpts), dtsb_idxs, 0], axis=0)  # Placeholder for the logit value
        # Compute the softmax distribution over the disturbance policies
        prob_un = np.exp(-self.softmax_rationality * logit)  # Negative here since dstb minimizes
        softmax_dist = prob_un / np.sum(prob_un) # Minus sign on logits since the better dtsb minimizes scores
        dtsb_idx = self.rng.choice(dtsb_idxs, p=softmax_dist)

        # If the chosen index is -1, use the dummy disturbance policy
        if dtsb_idx == -1:
            return self.dummy_dstb_policy
        else:
            # Otherwise, return the corresponding disturbance policy
            chosen_dstb = copy.deepcopy(self.dstb)
            chosen_dstb.restore(self.dstb_ckpts[dtsb_idx], self.model_folder, verbose=False)
            return chosen_dstb


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

        obsrv_all = obsrv_all.float().to(self.device)

        # Get control actions
        if self.cnt_step < self.warmup_steps:  # Warm up with random actions
            ctrl_action_all, _ = self.rnd_ctrl_policy.get_action(obsrv_all)
        else:
            with torch.no_grad():
                if self.ctrl.is_stochastic:
                    ctrl_action_all, _ = self.ctrl.sample(obsrv_all, append=None, latent=None)
                else:
                    ctrl_action_all = self.ctrl.net(obsrv_all, append=None, latent=None)
            ctrl_action_all = ctrl_action_all.cpu().numpy() # (num_envs, ctrl_action_dim)

        action_all = []
        dstb_sampler = self.dstb_sampler_list[0]

        with torch.no_grad():
            if dstb_sampler.is_stochastic:
                assert not isinstance(dstb_sampler, DummyPolicy), "Dummy policy cannot be stochastic."
                dstb_action, _ = dstb_sampler.sample(
                    obsrv_all[0], agents_action={"ctrl": ctrl_action_all[0]}, append=None, latent=None
                ) # (dstb_action_dim,)
            else:
                dstb_action, _ = dstb_sampler.get_action(
                    obsrv_all[0], agents_action={"ctrl": ctrl_action_all[0]}, append=None, latent=None
                ) # (dstb_action_dim,)
        if isinstance(dstb_action, torch.Tensor):
            dstb_action = dstb_action.cpu().numpy()
            
        action_all.append({'ctrl': ctrl_action_all[0], 'dstb': dstb_action})

        return action_all    
    
    
    def interact(
        self, rollout_env: Union[BaseZeroSumEnv, VecEnvBase], obsrv_all: torch.Tensor, action_all: List[Dict[str,
                                                                                                            np.ndarray]]
        ):
        """
        Executes one interaction step in the environment(s), stores transitions, and resets environment(s) if done. Tracks safety violations.

        Args:
            rollout_env: The environment to interact with.
            ovsrv_all: Current observation tensor for all environments.
            action_all: List of action dictionaries for each environment.
        
        Returns:
            obsrv_nxt_all: Next observation tensor for all environments.
        """

        if self.num_envs == 1:
            obsrv_nxt, r, done, info = rollout_env.step(action_all[0], cast_torch=True)
            obsrv_nxt_all = obsrv_nxt[None]
            r_all = np.array([r])
            done_all = np.array([done])
            info_all = np.array([info])
        else:
            obsrv_nxt_all, r_all, done_all, info_all = rollout_env.step(action_all)


        for env_idx, (done, info) in enumerate(zip(done_all, info_all)):
        # Stores the transition in memory. Note that `obsrv` and `action` are cpu tensors
            action = {k: torch.FloatTensor(v[None]) for k, v in action_all[env_idx].items()}
            self.store_transition(
                obsrv_all[[env_idx]].cpu(), action, r_all[env_idx], obsrv_nxt_all[[env_idx]].cpu(), done, info
            )

            if done:
                if self.num_envs == 1:
                    obsrv_nxt_all = rollout_env.reset(cast_torch=True)[None]
                else:
                    obsrv_nxt_all[env_idx] = rollout_env.reset_one(index=env_idx)
                g_x = info['g_x']
                if g_x < 0:
                    self.cnt_safety_violation += 1
                self.cnt_num_episode += 1
                self.dstb_sampler_list[env_idx] = self.get_dstb_sampler()

        # Updates records
        self.violation_record.append(self.cnt_safety_violation)
        self.episode_record.append(self.cnt_num_episode)

        # Updates counter
        self.cnt_step += self.num_envs
        self.cnt_opt_period += self.num_envs
        self.cnt_eval_period += self.num_envs
        
        return obsrv_nxt_all


    def update_one(
        self, batch: Batch, timer: int, update_ctrl: bool,
        update_dstb: bool = True
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        Performs one update step on the critic and (optionally) the actor networks.

        Args:
            batch (Batch): A batch of transitions from replay buffer.
            timer (int): Current global step or update counter.
            update_ctrl (bool): Whether to update the controller.
            update_dstb (bool): Whether to update the disturbance agent.

        Returns:
            Tuple of losses: 
                (critic_loss,
                ctrl_actor_loss, ctrl_entropy_loss, ctrl_alpha_loss,
                dstb_actor_loss, dstb_entropy_loss, dstb_alpha_loss)
        """

        ctrl_action = batch.action['ctrl']
        dstb_action = batch.action['dstb']

        # ---------------------- Critic Update ----------------------
        self.critic.net.train()
        self.critic.target.train()
        self.ctrl.net.eval()
        self.dstb.net.eval()

        with torch.no_grad():
            ctrl_action_nxt, _ = self.ctrl.sample(batch.non_final_obsrv_nxt)
            dstb_action_nxt, _ = self.dstb.sample(
                batch.non_final_obsrv_nxt,
                agents_action={"ctrl": ctrl_action_nxt.cpu().numpy()}
            )
        action = self.combine_action(ctrl_action, dstb_action)
        action_nxt = self.combine_action(ctrl_action_nxt, dstb_action_nxt)

        q1, q2 = self.critic.net(batch.obsrv, action)
        q1_nxt, q2_nxt = self.critic.target(batch.non_final_obsrv_nxt, action_nxt)

        
        loss_q = self.critic.update(
            q1=q1, q2=q2, q1_nxt=q1_nxt, q2_nxt=q2_nxt,
            non_final_mask=batch.non_final_mask,
            reward=batch.reward,
            g_x=batch.info['g_x'],
            l_x=batch.info['l_x'],
            binary_cost=batch.info['binary_cost']
        )

        # ------------------ Controller (Ctrl) Update ------------------
        if update_ctrl and timer % self.ctrl.update_period == 0:
            update_alpha = self.cnt_step >= self.warmup_steps

            self.ctrl.net.train()
            self.dstb.net.eval()
            self.critic.net.eval()

            ctrl_action_sample, log_prob = self.ctrl.sample(batch.obsrv)

            with torch.no_grad():
                if self.dstb.obsrv_list is None:
                    dstb_action_aux = self.dstb.net(batch.obsrv)
                else:
                    dstb_action_aux = self.dstb.net(batch.obsrv, action=ctrl_action_sample)

            action_sample = self.combine_action(ctrl_action_sample, dstb_action_aux)
            q1_sample, q2_sample = self.critic.net(batch.obsrv, action_sample)

            loss_ctrl, loss_ent_ctrl, loss_alpha_ctrl = self.ctrl.update(
                q1=q1_sample, q2=q2_sample,
                log_prob=log_prob,
                update_alpha=update_alpha
            )
        else:
            loss_ctrl = loss_ent_ctrl = loss_alpha_ctrl = 0.0

        # ----------------- Disturbance (Dstb) Update -----------------
        if update_dstb and timer % self.dstb.update_period == 0:
            update_alpha = self.cnt_step >= self.warmup_steps

            self.dstb.net.train()
            self.ctrl.net.eval()
            self.critic.net.eval()

            with torch.no_grad():
                ctrl_action_aux = self.ctrl.net(batch.obsrv)

            if self.dstb.obsrv_list is None:
                dstb_action_sample, log_prob = self.dstb.net.sample(batch.obsrv)
            else:
                dstb_action_sample, log_prob = self.dstb.net.sample(batch.obsrv, action=ctrl_action_aux)

            action_sample = self.combine_action(ctrl_action_aux, dstb_action_sample)
            q1_sample, q2_sample = self.critic.net(batch.obsrv, action_sample)

            loss_dstb, loss_ent_dstb, loss_alpha_dstb = self.dstb.update(
                q1=q1_sample, q2=q2_sample,
                log_prob=log_prob,
                update_alpha=update_alpha
            )
        else:
            loss_dstb = loss_ent_dstb = loss_alpha_dstb = 0.0

        # ------------------ Target Network Update ------------------
        if timer % self.critic.update_target_period == 0:
            self.critic.update_target()

        # Set all networks to eval mode again
        self.critic.net.eval()
        self.ctrl.net.eval()
        self.dstb.net.eval()

        return (
            loss_q,
            loss_ctrl, loss_ent_ctrl, loss_alpha_ctrl,
            loss_dstb, loss_ent_dstb, loss_alpha_dstb
        )


    def update(self):
        """
        Performs a joint update of the critic and (optionally) the controller and disturbance agents.
        Alternates between updating the controller and disturbance policies based on a set ratio.
        Tracks and logs various losses (Q-function, policy, entropy, alpha) and stores statistics.
        """

        # Check whether it's time to run an update cycle
        if self.cnt_step < self.min_steps_b4_opt or self.cnt_opt_period < self.opt_period:
            return  # Not enough steps yet

        # Reset optimization period counter
        self.cnt_opt_period = 0

        # Determine whether to update the controller
        update_ctrl = (self.cnt_dstb_updates == self.ctrl_update_ratio)
        if update_ctrl:
            self.cnt_dstb_updates = 0
            loss_ctrl_all, loss_ent_ctrl_all, loss_alpha_ctrl_all = [], [], []
        print(f"[Update] Step {self.cnt_step} | Update Controller: {update_ctrl}")

        # Initialize loss trackers
        loss_q_all = []
        loss_dstb_all, loss_ent_dstb_all, loss_alpha_dstb_all = [], [], []

        # Run multiple update steps
        for timer in range(self.num_updates_per_opt):
            # Try sampling a valid batch (with at least one non-terminal transition)
            for _ in range(10):
                batch = self.sample_batch()
                if torch.any(batch.non_final_mask):
                    break
            else:
                warnings.warn("Cannot get a valid batch after 10 attempts!", UserWarning)
                continue

            # Perform one joint update
            loss_q, loss_ctrl, loss_ent_ctrl, loss_alpha_ctrl, loss_dstb, loss_ent_dstb, loss_alpha_dstb = self.update_one(
                batch, timer, update_ctrl=update_ctrl
            )

            # Track critic and disturbance losses
            loss_q_all.append(loss_q)
            if timer % self.dstb.update_period == 0:
                loss_dstb_all.append(loss_dstb)
                loss_ent_dstb_all.append(loss_ent_dstb)
                loss_alpha_dstb_all.append(loss_alpha_dstb)

            # Track controller losses if it's being updated
            if update_ctrl and timer % self.ctrl.update_period == 0:
                loss_ctrl_all.append(loss_ctrl)
                loss_ent_ctrl_all.append(loss_ent_ctrl)
                loss_alpha_ctrl_all.append(loss_alpha_ctrl)

        # Compute loss means
        loss_q_mean = np.mean(loss_q_all)
        loss_dstb_mean = np.mean(loss_dstb_all)
        loss_ent_dstb_mean = np.mean(loss_ent_dstb_all)
        loss_alpha_dstb_mean = np.mean(loss_alpha_dstb_all)

        if update_ctrl:
            loss_ctrl_mean = np.mean(loss_ctrl_all)
            loss_ent_ctrl_mean = np.mean(loss_ent_ctrl_all)
            loss_alpha_ctrl_mean = np.mean(loss_alpha_ctrl_all)
        else:
            loss_ctrl_mean = loss_ent_ctrl_mean = loss_alpha_ctrl_mean = None

        # Store all loss statistics
        self.loss_record.append([
            loss_q_mean,
            loss_ctrl_mean, loss_ent_ctrl_mean, loss_alpha_ctrl_mean,
            loss_dstb_mean, loss_ent_dstb_mean, loss_alpha_dstb_mean
        ])

        # Logging (e.g., to wandb)
        if self.use_wandb:
            log_dict = {
                "loss/critic": loss_q_mean,
                "loss/dstb": loss_dstb_mean,
                "loss/entropy_dstb": loss_ent_dstb_mean,
                "loss/alpha_dstb": loss_alpha_dstb_mean,
                "metrics/cnt_safety_violation": self.cnt_safety_violation,
                "metrics/cnt_num_episode": self.cnt_num_episode,
                "hyper_parameters/alpha_ctrl": self.ctrl.alpha,
                "hyper_parameters/alpha_dstb": self.dstb.alpha,
                "hyper_parameters/gamma": self.critic.gamma,
            }
            if update_ctrl:
                log_dict.update({
                    "loss/ctrl": loss_ctrl_mean,
                    "loss/entropy_ctrl": loss_ent_ctrl_mean,
                    "loss/alpha_ctrl": loss_alpha_ctrl_mean,
                })
            wandb.log(log_dict, step=self.cnt_step, commit=False)

        # Increment disturbance update counter
        self.cnt_dstb_updates += 1


    def update_ctrl_agent(
        self, env: BaseZeroSumEnv, rollout_env: BaseZeroSumEnv, ctrl_ckpt_step: int
    ):
        """
        Updates the controller policy in the environment and rollout environment.

        Args:
            env (BaseZeroSumEnv): Environment used for evaluation.
            rollout_env (BaseZeroSumEnv): Environment used for rollouts.
            ctrl_ckpt_step (int): The checkpoint step of the controller.
        """

        # Load latest controller policy
        if ctrl_ckpt_step == self.cnt_step:
            self.ctrl_eval.update_policy(self.ctrl)
        else:
            self.ctrl_eval.restore(ctrl_ckpt_step, self.model_folder, verbose=False)

        # Update environment agents with the restored controller
        env.agent.policy.update_policy(self.ctrl_eval)
        rollout_env.agent.policy.update_policy(self.ctrl_eval)


    def update_leaderboard(self, eval_results: dict, ctrl_idx: int, dstb_idx: int):
        """
        Updates leaderboard with evaluation results for a specific (controller, disturbance) pair.
        """

        self.leaderboard[ctrl_idx, dstb_idx, 0] = eval_results[self.eval_metric]
        for metric_idx, aux_metric_name in enumerate(self.aux_metric):
            self.leaderboard[ctrl_idx, dstb_idx, 1 + metric_idx] = eval_results[aux_metric_name]


    def update_dstb_agent(self, dstb_ckpt_step: int):
        """
        Updates evaluation disturbance agent.
        """
        if dstb_ckpt_step == self.cnt_step:
            self.dstb_eval.update_policy(self.dstb)
        else:
            self.dstb_eval.restore(dstb_ckpt_step, self.model_folder, verbose=False)


    def prune_leaderboard(self):
        """
        Prunes the leaderboard by maintaining the top K controller and disturbance checkpoints.
        """

        # Always save critic checkpoints
        self.critic.save(self.cnt_step, self.model_folder)

        with np.printoptions(precision=3, suppress=False):
            print(self.leaderboard[..., 0])
        if len(self.ctrl_ckpts) == self.save_top_k_ctrl:
            ctrl_avg_metric = np.nanmean(self.leaderboard[..., 0], axis=1)
            ctrl_idx = np.argmin(ctrl_avg_metric) # Removes the ctrl ckpt that has the minimum average metric
            with np.printoptions(precision=3, suppress=False):
                print("ctrl results", ctrl_avg_metric)
            if ctrl_idx != self.save_top_k_ctrl:
                print(f'Saving current ctrl by removing {ctrl_idx}')
                self.ctrl.remove(self.ctrl_ckpts[ctrl_idx], self.model_folder)
                self.ctrl_ckpts[ctrl_idx] = self.cnt_step
                self.leaderboard[ctrl_idx] = self.leaderboard[-1]
                self.ctrl.save(self.cnt_step, self.model_folder)
        else:
            # Save the current controller checkpoint
            self.ctrl_ckpts.append(self.cnt_step)
            self.ctrl.save(self.cnt_step, self.model_folder)

        if len(self.dstb_ckpts) == self.save_top_k_dstb:
            dstb_avg_metric = np.nanmean(self.leaderboard[:, :-1, 0], axis=0)
            dstb_idx = np.argmax(dstb_avg_metric) # Removes the dstb ckpt that has the maximum average metric
            with np.printoptions(precision=3, suppress=False):
                print("dstb results", dstb_avg_metric)
            if dstb_idx != self.save_top_k_dstb:
                print(f'Saving current dstb by removin {dstb_idx}')
                self.dstb.remove(self.dstb_ckpts[dstb_idx], self.model_folder)
                self.dstb_ckpts[dstb_idx] = self.cnt_step
                self.leaderboard[:, dstb_idx] = self.leaderboard[:, -2]
                self.dstb.save(self.cnt_step, self.model_folder)
        else:
            self.dstb_ckpts.append(self.cnt_step)
            self.dstb.save(self.cnt_step, self.model_folder)
        print()
    
    
    def update_hyper_param(self):
        """
        Updates the hyperparameters of the controller, disturbance agent, and critic.
        """

        self.ctrl.update_hyper_param() # lr_pi, lr_alpha
        self.dstb.update_hyper_param() # lr_pi, lr_alpha
        flag_rst_alpha = self.critic.update_hyper_param() # lr_q, gamma
        if flag_rst_alpha:
            self.ctrl.reset_alpha()
            self.dstb.reset_alpha()
    
    
    def eval(
        self,
        env: BaseZeroSumEnv,
        rollout_env: Union[BaseZeroSumEnv, VecEnvBase],
        eval_callback,
        init_eval: bool = False
    ) -> bool:
        """
        Evaluates the current policies against saved checkpoints and log leaderboard metrics.

        Returns:
            bool: True if evaluation was performed, False otherwise.
        """
        if self.cnt_eval_period < self.eval_period and not init_eval:
            return False

        print(f"\n[Eval] Running evaluation at step {self.cnt_step}")
        self.cnt_eval_period = 0

        cur_ctrl_idx = len(self.ctrl_ckpts)
        cur_dstb_idx = len(self.dstb_ckpts)

        # === (1) Current disturbance vs. all controller checkpoints ===
        self.update_dstb_agent(dstb_ckpt_step=self.cnt_step)
        for ctrl_idx, ctrl_ckpt_step in enumerate(self.ctrl_ckpts):
            self.update_ctrl_agent(env, rollout_env, ctrl_ckpt_step=ctrl_ckpt_step)
            fig_path = os.path.join(self.figure_folder, f"{ctrl_ckpt_step}_{self.cnt_step}.png")
            eval_results = eval_callback(env=env, rollout_env=rollout_env, value_fn=self.value,
                                        adversary=self.dstb_eval, fig_path=fig_path)
            self.update_leaderboard(eval_results, ctrl_idx, cur_dstb_idx)

        # === (2) Current controller vs. all disturbance checkpoints ===
        self.update_ctrl_agent(env, rollout_env, ctrl_ckpt_step=self.cnt_step)
        for dstb_idx, dstb_ckpt_step in enumerate(self.dstb_ckpts):
            self.update_dstb_agent(dstb_ckpt_step=dstb_ckpt_step)
            fig_path = os.path.join(self.figure_folder, f"{self.cnt_step}_{dstb_ckpt_step}.png")
            eval_results = eval_callback(env=env, rollout_env=rollout_env, value_fn=self.value,
                                        adversary=self.dstb_eval, fig_path=fig_path)
            self.update_leaderboard(eval_results, cur_ctrl_idx, dstb_idx)

        # === (3) Current controller vs. current disturbance ===
        self.update_dstb_agent(dstb_ckpt_step=self.cnt_step)
        fig_path = os.path.join(self.figure_folder, f"{self.cnt_step}_{self.cnt_step}.png")
        eval_results = eval_callback(env=env, rollout_env=rollout_env, value_fn=self.value,
                                    adversary=self.dstb_eval, fig_path=fig_path)
        self.update_leaderboard(eval_results, cur_ctrl_idx, cur_dstb_idx)

        # === (4) Current controller vs. dummy disturbance ===
        fig_path = os.path.join(self.figure_folder, f"{self.cnt_step}_dummy.png")
        eval_results = eval_callback(env=env, rollout_env=rollout_env, value_fn=self.value,
                                    adversary=self.dummy_dstb_policy, fig_path=fig_path)
        self.update_leaderboard(eval_results, cur_ctrl_idx, -1)

        # === (5) Compute and log leaderboard summary ===
        log_dict = {
            f"eval/{self.eval_metric}_ctrl": np.nanmean(self.leaderboard[cur_ctrl_idx, :, 0]),
            f"eval/{self.eval_metric}_dstb": np.nanmean(self.leaderboard[:, cur_dstb_idx, 0])
        }
        for metric_idx, aux_metric_name in enumerate(self.aux_metric):
            log_dict[f"eval/{aux_metric_name}_ctrl"] = np.nanmean(
                self.leaderboard[cur_ctrl_idx, :, metric_idx + 1])
            log_dict[f"eval/{aux_metric_name}_dstb"] = np.nanmean(
                self.leaderboard[:, cur_dstb_idx, metric_idx + 1])

        self.eval_record.append(list(log_dict.values()))
        self.prune_leaderboard()

        if self.use_wandb:
            wandb.log(log_dict, step=self.cnt_step, commit=True)

        return True


    def save(self, max_model: Optional[int] = None):
        """
        Saves the current controller, disturbance, and critic models.
        """

        self.ctrl.save(self.cnt_step, self.model_folder, max_model)
        self.dstb.save(self.cnt_step, self.model_folder, max_model)
        self.critic.save(self.cnt_step, self.model_folder, max_model)


    def value(self, obsrv: np.ndarray, append: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the critic's value estimate for the current observation and action.
        
        Combines control and disturbance actions into a single tensor and passes it to the critic.
        
        Args:
            obsrv: Current observation array.
            append: Optional additional information appended to the critic.
        
        Returns:
            Critic's value estimate for the current observation and action.
        """

        obsrv_tensor = torch.FloatTensor(obsrv).to(self.device)
        with torch.no_grad():
            ctrl_action = self.ctrl.net(obsrv_tensor)
            dstb_action = self.dstb.net(obsrv_tensor)
        action = self.combine_action(ctrl_action, dstb_action)
        return self.critic.value(obsrv_tensor, action, append=append)


    def init_learn(self, env: BaseEnv) -> Union[BaseEnv, VecEnvBase]:
        """
        Initializes the learning process and resets internal counters.
        """

        self.cnt_dstb_updates = 0
        return super().init_learn(env)
