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
from typing import Optional
import torch
import wandb






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


