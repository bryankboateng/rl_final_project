# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for building blocks for actors and critics.

modified from: https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/replay_memory.py
"""

from typing import List
import numpy as np
import torch as th
from collections import deque, namedtuple
import os
import glob
import torch
import wandb
from typing import List, Callable, Any, Optional, Dict, Union
import numpy as np
import torch
from simulators.vec_env.vec_env import VecEnvBase
from simulators import BaseEnv, BaseZeroSumEnv
import pickle






def save_model(model: torch.nn.Module, step: int, model_folder: str, types: str, max_model: Optional[int] = None):
  start = len(types) + 1
  os.makedirs(model_folder, exist_ok=True)
  model_list = glob.glob(os.path.join(model_folder, '*.pth'))

  if max_model is not None:
    if len(model_list) > max_model - 1:
      min_step = min([int(li.split('/')[-1][start:-4]) for li in model_list])
      os.remove(os.path.join(model_folder, '{}-{}.pth'.format(types, min_step)))
  model_path = os.path.join(model_folder, '{}-{}.pth'.format(types, step))
  torch.save(model.state_dict(), model_path)



# required by train_isaac
def evaluate_zero_sum(
    env: BaseZeroSumEnv, rollout_env: Union[BaseZeroSumEnv, VecEnvBase],
    adversary: Callable[[np.ndarray, np.ndarray, Any],
                        np.ndarray], value_fn: Callable[[np.ndarray, Optional[Union[np.ndarray, torch.Tensor]]],
                                                        np.ndarray], num_trajectories: int, end_criterion: str,
    timeout: int, reset_kwargs_list: Optional[Union[List[Dict], Dict]] = None,
    rollout_step_callback: Optional[Callable] = None, rollout_episode_callback: Optional[Callable] = None,
    visualize_callback: Optional[Callable] = None, fig_path: Optional[str] = None, **kwargs
) -> Dict:

  if isinstance(rollout_env, BaseZeroSumEnv):
    _, results, length = rollout_env.simulate_trajectories(
        num_trajectories=num_trajectories, T_rollout=timeout, end_criterion=end_criterion, adversary=adversary,
        reset_kwargs_list=reset_kwargs_list, rollout_step_callback=rollout_step_callback,
        rollout_episode_callback=rollout_episode_callback, use_tqdm=True, **kwargs
    )
  else:
    _, results, length = rollout_env.simulate_trajectories_zs(
        num_trajectories=num_trajectories, T_rollout=timeout, end_criterion=end_criterion, adversary=adversary,
        reset_kwargs_list=reset_kwargs_list, rollout_step_callback=rollout_step_callback,
        rollout_episode_callback=rollout_episode_callback, use_tqdm=True, **kwargs
    )

  safe_rate = np.sum(results != -1) / num_trajectories
  info = {"safety": safe_rate, "ep_length": np.mean(length)}
  if end_criterion == "reach-avoid":
    ra_rate = np.sum(results == 1) / num_trajectories
    info["reach-avoid"] = ra_rate

  if visualize_callback is not None:
    visualize_callback(env, value_fn, adversary=adversary, fig_path=fig_path)
  return info


def get_model_index(parent_dir, model_type, step: Optional[int] = None, type="mean_critic", cutoff=0, autocutoff=None):
  """_summary_

  Args:
      parent_dir (str): dir of where the result is
      model_type (str): type of model ("highest", "safest", "worst", "manual")
      step (int, optional): step to load in manual
      type (str, optional): model type to read index from [mean_critic, adv_critic, dstb, ctrl]. Defaults to "mean_critic".
      cutoff (int, optional): Model cutoff, will not consider any model index that's lower than this value. Defaults to None.
      autocutoff (float, optional): Auto calculate where to cutoff by taking percentage of the horizon. Defaults to None

  Raises:
      ValueError: _description_

  Returns:
      _type_: _description_
  """
  print("WARNING: using model index stored in {}".format(type))
  model_dir = os.path.join(parent_dir, "model", type)
  print(model_dir)
  chosen_run_iter = None
  print("\tModel type: {}".format(model_type))

  if model_type == "highest":
    model_list = os.listdir(model_dir)
    highest_number = sorted([int(a.split("-")[1].split(".")[0]) for a in model_list])[-1]
    print("\t\tHighest training number: {}".format(highest_number))
    chosen_run_iter = highest_number

  elif model_type == "safest" or model_type == "worst":
    # get the run with the best result from train.pkl
    train_log_path = os.path.join(parent_dir, "train.pkl")
    with open(train_log_path, "rb") as log:
      train_log = pickle.load(log)
    data = np.array(sorted(train_log["pq_top_k"]))
    if autocutoff is not None:
      print("\t\t\tAuto cutting off with {} ratio".format(autocutoff))
      cutoff = max(data[:, 1]) * autocutoff
    data = data[data[:, 1] > cutoff]
    print("\t\t\tTaking {} of max value {}".format(autocutoff, max(data[:, 1])))

    if model_type == "safest":
      safe_rate, best_iteration = data[-1]
      print("\t\tBest training iteration: {}, safe rate: {}".format(best_iteration, safe_rate))
    else:
      safe_rate, best_iteration = data[0]
      print("\t\tWorst training iteration: {}, safe rate: {}".format(best_iteration, safe_rate))
    chosen_run_iter = best_iteration

  elif model_type == "manual":
    model_list = os.listdir(model_dir)
    iter_list = [int(a.split("-")[1].split(".")[0]) for a in model_list]
    if step in iter_list:
      print("\t\tManual pick a model iteration: {}".format(step))
      chosen_run_iter = step
    else:
      raise ValueError("Cannot find iteration {} in list of runs".format(step))

  assert chosen_run_iter is not None, "Something is wrong, cannot find the chosen run, check evaluation config"

  return int(chosen_run_iter), os.path.join(parent_dir, "model")


class _scheduler(object):

  def __init__(self, last_epoch=-1, verbose=False):
    self.cnt = last_epoch
    self.verbose = verbose
    self.variable = 0.0  # Defaults to 0.0. It will be overwritten by the first call to step().
    self.step()

  def step(self):
    """Increments counter and updates variable value."""
    self.cnt += 1
    value = self.get_value()
    self.variable = value

  def get_value(self):
    """Defines how the scheduled value changes over time. To be overridden by subclasses."""
    raise NotImplementedError

  def get_variable(self) -> float:
    """Returns the current scheduled variable value."""
    return self.variable


class StepLRMargin(_scheduler):
  """Step-based scheduler that interpolates between an initial and goal value.

  Instead of direct decay, this scheduler shifts the variable from `init_value`
  to `goal_value` over time, applying decay to the difference.

  Attributes:
      init_value (float): Initial value.
      goal_value (float): Final target value.
      period (int): Number of steps before applying decay.
      decay (float): Decay factor.
      end_value (float or None): Maximum change limit.
      threshold (int): Delay before applying decay.
  """
  def __init__(
      self, init_value, period, goal_value, decay=0.1, end_value=None, last_epoch=-1, threshold=0, verbose=False
  ):
    self.init_value = init_value
    self.period = period
    self.decay = decay
    self.end_value = end_value
    self.goal_value = goal_value
    self.threshold = threshold
    super(StepLRMargin, self).__init__(last_epoch, verbose)

  def get_value(self):
    cnt = self.cnt - self.threshold
    if cnt < 0:
      return self.init_value

    numDecay = int(cnt / self.period)
    # Different tmpValue
    tmpValue = self.goal_value - (self.goal_value - self.init_value) * (self.decay**numDecay)
    if self.end_value is not None and tmpValue >= self.end_value:
      return self.end_value
    return tmpValue



# `Transition` is a named tuple representing a single transition in our
# RL environment. All the other information is stored in the `info`, e.g.,
# `g_x`, `l_x`, `binary_cost`, `append`, and `latent`, etc. Note that we also
# require all the values to be np.ndarray or float. Here `a` is a dictionary.

# Batch objects built on transitions with reward, observations, valid_next_obs(with masking), actions and info batched
Transition = namedtuple('Transition', ['s', 'a', 'r', 's_', 'done', 'info'])


class Batch(object):

  def __init__(self, transitions: List[Transition], device: th.device):
    self.device = device
    batch = Transition(*zip(*transitions)) #Use unpacking and zipping for batching

    # Reward and Done.
    self.reward = th.FloatTensor(batch.r).to(device)
    self.non_final_mask = th.BoolTensor(np.logical_not(np.asarray(batch.done))).to(device)

    # obsrv.
    self.non_final_obsrv_nxt = th.cat(batch.s_)[self.non_final_mask].to(device) # Valid next observtions for reward-to-go
    self.obsrv = th.cat(batch.s).to(device)

    # Action.
    #(key; torch(a1,a2,a3,...))
    self.action = {}
    for key in batch.a[0].keys():
      self.action[key] = th.cat([a[key] for a in batch.a]).to(device)

    # Info.
    self.info = {}
    for key, value in batch.info[0].items():
      if isinstance(value, np.ndarray) or isinstance(value, float):
        self.info[key] = th.FloatTensor(np.asarray([info[key] for info in batch.info])).to(device)
    if 'append' in self.info:
      self.info['non_final_append_nxt'] = (batch.info['append_nxt'][self.non_final_mask])

# Replay Buffer: Intialize, push, sample
class ReplayMemory(object):

  def __init__(self, capacity, seed):
    self.reset(capacity)
    self.capacity = capacity
    self.seed = seed
    self.rng = np.random.default_rng(seed=self.seed)

  def reset(self, capacity):
    if capacity is None:
      capacity = self.capacity
    self.memory = deque(maxlen=capacity)

  def update(self, transition):
    self.memory.appendleft(transition)  # pop from right if full

  def sample(self, batch_size):
    length = len(self.memory)
    indices = self.rng.integers(low=0, high=length, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def sample_recent(self, batch_size, recent_size):
    recent_size = min(len(self.memory), recent_size)
    indices = self.rng.integers(low=0, high=recent_size, size=(batch_size,))
    return [self.memory[i] for i in indices]

  def __len__(self):
    return len(self.memory)


