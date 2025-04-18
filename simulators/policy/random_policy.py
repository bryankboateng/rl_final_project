from typing import Tuple, Union, Dict, Optional
import numpy as np
import torch
import time

from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
  policy_type = "random"

  def __init__(self, id: str, action_range: np.ndarray, seed: int) -> None:
    super().__init__(id)
    self.action_range = np.array(action_range, dtype=np.float32)
    self.rng = np.random.default_rng(seed=seed)

  @property
  def is_stochastic(self) -> bool:
    return True

  def get_action(
      self, obsrv: Union[np.ndarray, torch.Tensor], agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    if obsrv.ndim == 1:
      size = self.action_range.shape[0] #action_range --> (dim_action, 2) : [[a1_min,a1_max], [a2_min, a2_max]]
    else:
      size = (obsrv.shape[0], self.action_range.shape[0]) #--> (num_obs, dim_action)

    time0 = time.time()
    action = self.rng.uniform(low=self.action_range[:, 0], high=self.action_range[:, 1], size=size)
    t_process = time.time() - time0
    status = 1
    return action, dict(t_process=t_process, status=status)

  def sample(
      self, obsrv: Union[np.ndarray, torch.Tensor], agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    size = self.action_range.shape[0]
    action = self.rng.uniform(low=self.action_range[:, 0], high=self.action_range[:, 1], size=size)
    log_prob_one = np.sum(np.log(1 / (self.action_range[:, 1] - self.action_range[:, 0])))

    if isinstance(obsrv, torch.Tensor):
      action = torch.from_numpy(action).to(obsrv.device)
      log_prob = torch.full(size=(size,), fill_value=log_prob_one, device=obsrv.device)
    else:
      log_prob = np.full(shape=(size,), fill_value=log_prob_one)

    return action, log_prob
