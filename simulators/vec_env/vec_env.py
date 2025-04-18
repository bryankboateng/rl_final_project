# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for vectorized environments especially for RL training.

Modified from stable-baseline3 and https://github.com/allenzren/alano.
"""

from typing import Type, Any, Dict, Optional
import torch as th
import gym
from .subproc_vec_env import SubprocVecEnv


def make_vec_envs(
    env_type: Any, num_processes: int, seed: int = 0, cpu_offset: int = 0,
    vec_env_type: Type[SubprocVecEnv] = SubprocVecEnv, venv_kwargs: Optional[Dict] = None,
    env_kwargs: Optional[Dict] = None
) -> SubprocVecEnv:

  if env_kwargs is None:
    env_kwargs = {}
  if venv_kwargs is None:
    venv_kwargs = {}
  if isinstance(env_type, str):
    envs = [gym.make(env_type, **env_kwargs) for i in range(num_processes)]
  else:
    envs = [env_type(**env_kwargs) for _ in range(num_processes)]
  for rank, env in enumerate(envs):
    env.seed(seed + rank)
  envs = vec_env_type(envs, cpu_offset, **venv_kwargs)
  return envs


class VecEnvBase(SubprocVecEnv):
  """
  Mostly for torch
  """

  def __init__(
      self, venv, cpu_offset=0, device: str = th.device("cpu"), pickle_option='cloudpickle', start_method=None
  ):
    super(VecEnvBase, self).__init__(venv, cpu_offset, pickle_option=pickle_option, start_method=start_method)
    self.device = device

  def reset(self, **kwargs):
    obsrv = super().reset(**kwargs)
    return th.FloatTensor(obsrv).to(self.device)

  def reset_one(self, index, **kwargs):
    obsrv = self.env_method('reset', indices=[index], **kwargs)[0]
    return th.FloatTensor(obsrv).to(self.device)

  # Overrides
  def step_async(self, actions):
    if isinstance(actions, th.Tensor):
      actions = actions.cpu().numpy()
    super().step_async(actions)

  # Overrides
  def step_wait(self):
    obsrv, reward, done, info = super().step_wait()
    obsrv = th.FloatTensor(obsrv).to(self.device)
    reward = th.FloatTensor(reward).unsqueeze(dim=1).float()
    return obsrv, reward, done, info

  def get_obsrv(self, states):
    obsrv = super().get_obsrv(states)
    return th.FloatTensor(obsrv).to(self.device)
