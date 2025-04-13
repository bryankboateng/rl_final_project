# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Classes for building blocks for actors and critics.

modified from: https://github.com/SafeRoboticsLab/SimLabReal/blob/main/agent/model.py
"""

from typing import Optional, Union, Tuple, List
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Optional, Union, Tuple, Dict, List
from abc import ABC, abstractmethod
import os
from torch.optim import Adam, AdamW
from simulators.policy.base_policy import BasePolicy
from utils import save_model, StepLRMargin
from torch.optim.lr_scheduler import StepLR

activation_dict = nn.ModuleDict({
    "ReLU": nn.ReLU(),
    "Tanh": nn.Tanh(),
    "Identity": nn.Identity()
})


####### Design of MLP layers for downstream architectures #######

class MLP(nn.Module):
  """
  Construct a fully-connected neural network with flexible depth, width and
  activation function choices.
  """

  def __init__(
      self, dim_list: list, activation_type: str = 'Tanh', out_activation_type: str = 'Identity', verbose: bool = False
  ):
    """Initalizes the multilayer Perceptrons.

    Args:
        dim_list (list of integers): the dimension of each layer.
        activation_type (str, optional): type of activation. Support 'Sin', 'Tanh' and 'ReLU'. Defaults to 'Tanh'.
        out_activation_type (str, optional): type of output activation. Support 'Sin', 'Tanh' and 'ReLU'. Defaults to
            'Tanh'.
        use_ln (bool, optional): uses layer normalization or not. Defaults to False.
        use_spec (bool, optional): uses spectral normalization or not. Defaults to False.
        use_bn (bool, optional): uses batch normalization or not. Defaults to False.
        verbose (bool, optional): prints info or not. Defaults to False.
    """
    super(MLP, self).__init__()
    self.moduleList = nn.ModuleList()
    numLayer = len(dim_list) - 1
    for idx in range(numLayer):
      i_dim = dim_list[idx]
      o_dim = dim_list[idx + 1]

      linear_layer = nn.Linear(i_dim, o_dim)
      module = nn.Sequential(
        OrderedDict([
            ('linear_1', linear_layer),
            ('act_1', activation_dict[activation_type]),
        ])
        )
      if idx == numLayer - 1:
        module = nn.Sequential(
            OrderedDict([
                ('linear_1', linear_layer),
                ('act_1', activation_dict[out_activation_type]),
            ])
        )
      self.moduleList.append(module)
    if verbose:
        print(self.moduleList)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for m in self.moduleList:
      x = m(x)
    return x


def tie_weights(src, trg):
  assert type(src) is type(trg)
  trg.weight = src.weight
  trg.bias = src.bias


def get_mlp_input(
    obsrv: Union[np.ndarray, torch.Tensor],
    device: torch.device,
    action: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, bool, int]:
    """
    Converts inputs (possibly numpy arrays) to torch tensors on the given device
    and concatenates them. Also tracks whether the input was originally numpy
    (np_input) and how many dimensions were artificially expanded (num_extra_dim).

    Args:
        obsrv: Observation data (np.ndarray or torch.Tensor).
        device: Target device for new tensor.
        action: Action data (optional).


    Returns:
        (processed_input, np_input, num_extra_dim)
    """
    # HINT:
    # 1. Check if obsrv is numpy => convert to torch FloatTensor => move to device.
    # 2. Do the same for action
    # 3. Possibly unsqueeze(0) if input has shape [dim] instead of [batch, dim].
    # 4. Concat them along last dimension (e.g. dim=-1).
    # 5. Keep track if obsrv was originally numpy (np_input = True/False).
    # 6. Keep track of how many dimensions you had to add (num_extra_dim).
    # 7. Return (processed_input, np_input, num_extra_dim).
    if isinstance(obsrv, np.ndarray):
        obsrv = torch.FloatTensor(obsrv).to(device)
        np_input = True
        
        
    if len(obsrv.shape) == 1:
        obsrv = obsrv.unsqueeze(0)
        num_extra_dim = 1
        
    if action is not None:
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(device)
        obsrv = torch.concat((obsrv, action), dim=-1)
    
    
        
    
    return obsrv, np_input, num_extra_dim


########################################################
# Twinned Q-network
########################################################

class TwinnedQNetwork(nn.Module):
    """
    A pair of Q-networks (Q1 and Q2) that map (obsrv + action) -> Q-value.
    Typically used in off-policy algorithms to reduce overestimation bias.
    """


    def __init__(
        self,
        obsrv_dim: int,
        mlp_dim: List[int],
        action_dim: int,
        activation_type: str = 'Tanh',
        device: Union[str, torch.device] = 'cpu',
        verbose: bool = True
    ):
        """
        Args:
            obsrv_dim: Dimension of observation input.
            mlp_dim: List of hidden-layer sizes (e.g. [64, 64]).
            action_dim: Dimension of the action input.
            activation_type: String identifier for activation (e.g. "Tanh").
            device: Which device to use, e.g. "cpu" or "cuda".
            verbose: Whether to print debugging info about the network.
        """
        super(TwinnedQNetwork, self).__init__()

        # HINT:
        # 1. Build MLP for self.Q1 using dimension list [obsrv_dim + action_dim, ... mlp_dim ..., 1].
        # 2. Duplicate it for self.Q2 (e.g. copy.deepcopy).
        # 3. Move networks to cuda if device is appropriate and store device as self.device = torch.device(device).
        # 4. If verbose, print layer info, etc.
        
        self.Q1 = MLP([obsrv_dim + action_dim] + mlp_dim + [1], activation_type=activation_type, out_activation_type='Identity', verbose=verbose)
        self.Q2 = copy.deepcopy(self.Q1)
        if device == 'cuda':
            self.Q1.to(device)
            self.Q2.to(device)
        self.device = torch.device(device)
        if verbose:
            print(f"Q1: {self.Q1}")
            print(f"Q2: {self.Q2}")


    def forward(
        self,
        obsrv: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Forward pass for both Q-networks Q1 and Q2.

        Args:
            obsrv: Observation input (numpy or torch.Tensor).
            action: Action input (numpy or torch.Tensor).

        Returns:
            (q1, q2): Q-values from the first and second networks respectively.
        """
        # Explanation of dimension maintenance & "numpiness":
        # 1) We use get_mlp_input() to handle device placement and dimension shape.
        # 2) We track if the input was originally numpy (np_input).
        # 3) We track if we artificially expanded dims (num_extra_dim).
        #
        # After forward pass, we can "undo" those expansions with squeeze()
        # and convert back to np if needed.

        processed_input, np_input, num_extra_dim = get_mlp_input(obsrv, self.device, action)
        q1 = self.Q1(processed_input)
        q2 = self.Q2(processed_input)
        if num_extra_dim > 0:
            q1 = q1.squeeze(0)
            q2 = q2.squeeze(0)
        if np_input:
            q1 = q1.detach().cpu().numpy()
            q2 = q2.detach().cpu().numpy()
        
        return q1, q2



########################################################
# Gaussian Policy
########################################################

class GaussianPolicy(nn.Module):
    """
    Outputs a mean and log_std for a Gaussian distribution. A tanh transform
    maps samples into the valid action range [a_min, a_max].
    """

    def __init__(
        self,
        obsrv_dim: int,
        mlp_dim: List[int],
        action_dim: int,
        action_range: np.ndarray,
        activation_type: str = 'Tanh',
        device: Union[str, torch.device] = 'cpu',
        verbose: bool = True
    ):
        """
        Args:
            obsrv_dim: Dimension of observations.
            mlp_dim: Hidden-layer sizes for the networks (list of ints).
            action_dim: Number of output action dimensions.
            action_range: np.ndarray of shape (action_dim, 2), specifying [min, max].
            append_dim: Extra features dimension to be concatenated with obsrv (if any).
            latent_dim: Dimension of any latent variables (if any).
            activation_type: Activation function for hidden layers.
            device: "cpu" or "cuda".
            verbose: Whether to print network info.
        """
        super().__init__()

        # HINT:
        # 1. Build self.mean MLP => dimension list = [obsrv_dim + append_dim + latent_dim, ... mlp_dim ..., action_dim].
        # 2. Build self.log_std MLP => same dimension list as above.
        # 3. Convert action_range to torch, compute self.scale & self.bias => for mapping [-1,1] to [min,max].
        # 4. Set self.LOG_STD_MAX, self.LOG_STD_MIN, self.eps for numeric stability.
        # 5. Store device.
        
        self.mean = MLP([obsrv_dim] + mlp_dim + [action_dim], activation_type=activation_type, out_activation_type='Identity', verbose=verbose)
        self.log_std = MLP([obsrv_dim] + mlp_dim + [action_dim], activation_type=activation_type, out_activation_type='Identity', verbose=verbose)
        
        self.action_dim = action_dim
        self.action_range = action_range
        self.device = torch.device(device)
        self.a_min = torch.FloatTensor(action_range[:, 0])
        self.a_max = torch.FloatTensor(action_range[:, 1])
        self.scale = (self.a_max - self.a_min) / 2
        self.bias = (self.a_max + self.a_min) / 2
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20
        self.eps = 1e-6
        
        


    def forward(
        self,
        obsrv: Union[np.ndarray, torch.Tensor],

    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Computes a deterministic action via the 'mean' network.

        Args:
            obsrv: Observation input (np or torch).
            action: Possibly used to incorporate other agents' actions.
            append: Additional data to be concatenated.
            latent: Latent variables.

        Returns:
            The action in environment scale (possibly as np if original input was np).
        """
        # Explanation of dimension maintenance & "numpiness":
        # 1) We call get_mlp_input to do dimension checks and concatenation.
        # 2) We apply the mean MLP => produce an "unbounded" action.
        # 3) Tanh => scale => shift => final action in [a_min, a_max].
        # 4) Possibly squeeze dimension if it was artificially added.
        # 5) Convert to np if input was originally np.

        # 1. obsrv_cat, np_input, num_extra_dim = get_mlp_input(obsrv, self.device, action, append, latent)
        # 2. a_unbounded = self.mean(obsrv_cat)
        # 3. a_tanh = torch.tanh(a_unbounded)
        # 4. scaled_action = a_tanh * self.scale + self.bias
        # 5. Possibly squeeze dimension.
        # 6. Possibly convert to numpy.
        # 7. return scaled_action
        
        processed_input, np_input, num_extra_dim = get_mlp_input(obsrv, self.device)
        a_unbounded = self.mean(processed_input)
        
        a_tanh = torch.tanh(a_unbounded)
        scaled_action = a_tanh * self.scale + self.bias
        if num_extra_dim > 0:
            scaled_action = scaled_action.squeeze(0)
        if np_input:
            scaled_action = scaled_action.detach().cpu().numpy()
        return scaled_action



    def sample(
        self,
        obsrv: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Samples an action stochastically using mean & log_std. Also returns log_prob.

        Args:
            obsrv: Observation input (np or torch).
            action: Possibly used for multi-agent settings.
            append: Additional data to be concatenated.
            latent: Latent variables.

        Returns:
            (sampled_action, log_prob)
        """
        # Explanation of dimension maintenance & "numpiness":
        # 1) Use get_mlp_input => merges obsrv+action+append+latent => track batch shape.
        # 2) Evaluate mean, log_std => clamp log_std => create Normal distribution => reparameterize => x = mean + std*N(0,1).
        # 3) y = tanh(x), then scale to [a_min, a_max].
        # 4) log_prob = log p(x) - log|det da/dx|.
        # 5) Possibly squeeze dimension & convert to np if needed.

        # HINT:
        # 1. obsrv_cat, np_input, num_extra_dim = get_mlp_input(...)
        # 2. mean_val = self.mean(obsrv_cat)
        # 3. log_std_val = self.log_std(obsrv_cat)
        # 4. clamp log_std_val between LOG_STD_MIN, LOG_STD_MAX
        # 5. x = Normal(mean_val, exp(log_std_val)).rsample()
        # 6. y = tanh(x), action = y * self.scale + self.bias
        # 7. Compute log_prob. Sum over action dimensions.
        # 8. Squeeze dimension if needed. Convert to np if needed.
        # 9. return (action, log_prob)
        
        processed_input, np_input, num_extra_dim = get_mlp_input(obsrv, self.device)
        mean_val = self.mean(processed_input)
        log_std_val = self.log_std(processed_input)
        log_std_val = torch.clamp(log_std_val, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        dist = torch.distributions.Normal(mean_val, torch.exp(log_std_val))
        xt = dist.rsample()
        yt = torch.tanh(xt)  # y = tanh(x) --> p(y) = p(tanh(x)) * |d(tanh(x))/dx|^(-1) transformation rule for probability
        action = self.scale * yt + self.bias
        log_prob = dist.log_prob(xt)
        log_prob -= torch.log(self.max_action * (1 - yt.pow(2)) + self.eps)  #d[tanh(x)]/dx = sech^2(x) = 1 - tanh^2(x)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
        



    def to(self, device: Union[str, torch.device]):
        """
        Moves module + any buffers/tensors (like action range) to a new device.
        """
        super().to(device)
        # HINT:
        # 1. self.device = device
        # 2. Move self.a_max, self.a_min, self.scale, self.bias => device
        
        self.device = device
        self.a_min = self.a_min.to(device)
        self.a_max = self.a_max.to(device)
        self.scale = self.scale.to(device)
        self.bias = self.bias.to(device)


    @property
    def is_stochastic(self) -> bool:
        """
        Indicates whether this policy outputs stochastic actions.
        """
        return True






### Skeleton class for managing hyperparams, model persistence, and optimizers ###

class BaseBlock(ABC):
  net: torch.nn.Module # class-level type hinting/attr #Can be explicitly overridden in subclass

  def __init__(self, cfg, device: torch.device) -> None:
    self.eval = cfg.eval
    self.device = device
    self.net_name: str = cfg.net_name

  @abstractmethod
  def build_network(self, verbose: bool = True):
    raise NotImplementedError

  def build_optimizer(self, cfg): #builds a scheduled optimizer
    
    # Choose optimizer type based on cf
    if cfg.opt_type == "AdamW":
      self.opt_cls = AdamW
    elif cfg.opt_type == "Adam":
      self.opt_cls = Adam
    else:
      raise ValueError("Not supported optimizer type!")

    # Learning Rate
    # Decide between fixed or scheduled learning rate
    self.lr_schedule: bool = cfg.lr_schedule
    if self.lr_schedule:
      self.lr_period = int(cfg.lr_period)
      self.lr_decay = float(cfg.lr_decay)
      self.lr_end = float(cfg.lr_end)
    self.lr = float(cfg.lr)

    # Builds the optimizer.
    # if scheduled then wraps optimizer in scheduler
    self.optimizer = self.opt_cls(self.net.parameters(), lr=self.lr, weight_decay=0.01)
    if self.lr_schedule:
      self.scheduler = StepLR(self.optimizer, step_size=self.lr_period, gamma=self.lr_decay)

  def update_hyper_param(self): #only valid during training and if scheduled
    #optimizers maintains a state dict with (state,...) (param_group,...)
    if not self.eval:
      if self.lr_schedule:
        lr = self.optimizer.state_dict()['param_groups'][0]['lr'] #param_group:list of dicts 
        if lr <= self.lr_end: #prevents lr from decaying past stopping point
          for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_end
        else:
          self.scheduler.step()

  # Manages model persistence (saving, loading, deleting)
  def save(self, step: int, model_folder: str, max_model: Optional[int] = None, verbose: bool = True) -> str:
    path = os.path.join(model_folder, self.net_name)
    save_model(self.net, step, path, self.net_name, max_model)
    if verbose:
      print(f"  => Saves {self.net_name} at {path}.")
    return path

  def restore(self, step: int, model_folder: str, verbose: bool = True) -> str:
    path = os.path.join(model_folder, self.net_name, f'{self.net_name}-{step}.pth')
    self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
    self.net.to(self.device)
    if verbose:
      print(f"  => Restores {self.net_name} at {path}.")
    return path

  def remove(self, step: int, model_folder: str, verbose: bool = True) -> str:
    path = os.path.join(model_folder, self.net_name, f'{self.net_name}-{step}.pth')
    if verbose:
      print(f"  => Removes {self.net_name} at {path}.")
    if os.path.exists(path):
      os.remove(path)
    return path


####### Design for ACTOR and CRITIC #######


class Actor(BaseBlock, BasePolicy):
    """
    High-level policy block that manages GaussianPolicy and hyperparameters like alpha.
    Inherits from BaseBlock (for config/device handling) and BasePolicy (for policy-like API).
    """

    # Class-level attributes:
    policy_type: str = "NNCS"  # e.g., a label for the type of policy
    net: GaussianPolicy        # or placeholder for other policy architectures (e.g., GMM)


    def __init__(
        self,
        cfg,
        cfg_arch,
        device: torch.device,
        obsrv_list: Optional[List] = None,
        verbose: bool = True
    ) -> None:
        """
        Args:
            cfg: Configuration for the learning/training process (e.g., alpha, min_alpha).
            cfg_arch: Architecture config (e.g., obsrv_dim, mlp_dim, action_dim).
            device: Torch device (cpu or cuda).
            obsrv_list: (Optional) Some list of observation indices or agent IDs for multi-agent synergy.
            verbose: If True, prints extra debug info.
        """
        # Initialize superclasses:
        BaseBlock.__init__(self, cfg, device)
        BasePolicy.__init__(self, id=self.net_name, obsrv_list=obsrv_list)

        # HINT:
        # 1. Define action_dim, action_range, actor_type from cfg_arch/cfg.
        # 2. Possibly define self.update_period, if not in evaluation mode.
        # 3. Call self.build_network(...) with relevant params.
        self.action_dim = cfg_arch.action_dim
        self.action_range = cfg_arch.action_range
        self.actor_type = cfg_arch.actor_type
        
        if not self.eval:
            self.update_period = cfg.update_period
            
        self.build_network(cfg, cfg_arch, verbose=verbose)



    @property
    def is_stochastic(self) -> bool:
        """
        Indicates if the underlying net is a stochastic policy.
        """
        return self.net.is_stochastic



    @property
    def alpha(self) -> torch.Tensor:
        """
        Returns the current entropy coefficient alpha as exp(log_alpha).
        """
        # HINT:
        return self.log_alpha.exp()



    def build_network(self, cfg, cfg_arch, verbose: bool = True):
        """
        Builds the policy network(s) based on the config, e.g., GaussianPolicy.
        Also handles loading pretrained weights if cfg_arch.pretrained_path is set.
        Puts network in eval mode if self.eval == True.
        """
        # HINT:
        # 1. Create self.net = GaussianPolicy(...)
        # 2. If there's a 'pretrained_path', load network weights except log_std.
        # 3. If self.eval: set net.eval(), freeze parameters, define self.log_alpha, etc.
        #    else: call self.build_optimizer(cfg)
        self.net = GaussianPolicy(
            obsrv_dim=cfg_arch.obsrv_dim,
            mlp_dim=cfg_arch.mlp_dim,
            action_dim=cfg_arch.action_dim,
            action_range=self.action_range,
            activation_type=cfg_arch.activation_type,
            device=self.device,
            verbose=verbose
        )
        if self.eval:
            self.net.eval() #what happens when you put a model on self_eval?
            for _, param in self.net.named_parameters():
                param.requires_grad = False
            self.log_alpha = torch.log(torch.FloatTensor([1e-8])).to(self.device)
        else:
            self.build_optimizer(cfg)


    def build_optimizer(self, cfg):
        super().build_optimizer(cfg)

        # entropy-related parameters
        
        #so entropy coefficient is allowed to vary 
        self.init_alpha = torch.log(torch.FloatTensor([cfg.alpha])).to(self.device)
        self.min_alpha = torch.log(torch.FloatTensor([cfg.min_alpha])).to(self.device)
        self.learn_alpha: bool = cfg.learn_alpha
        self.log_alpha = self.init_alpha.detach().clone()
        #Target entropy is defined using dimension of action in order for entropy to naturally scale with more dimensions
        self.target_entropy = torch.tensor(-self.action_dim).to(self.device)
        # are we learning the entropy coefficient with optimizer and scheduler?? Yes
        if self.learn_alpha:
            self.log_alpha.requires_grad = True
            self.lr_al: float = cfg.lr_al
            self.lr_al_schedule: bool = cfg.lr_al_schedule
            self.log_alpha_optimizer = self.opt_cls([self.log_alpha], lr=self.lr_al, weight_decay=0.01)
        if self.lr_al_schedule:
            self.lr_al_period: int = cfg.lr_al_period
            self.lr_al_decay: float = cfg.lr_al_decay
            self.lr_al_end: float = cfg.lr_al_end
            self.log_alpha_scheduler = StepLR(
                self.log_alpha_optimizer, step_size=self.lr_al_period, gamma=self.lr_al_decay
            )


    def update_hyper_param(self):
        if not self.eval:
         super().update_hyper_param()

        # Define update rule for the learning rate of entropy coeff?
        if self.learn_alpha and self.lr_al_schedule:
            lr = self.log_alpha_optimizer.state_dict()['param_groups'][0]['lr']
            if lr <= self.lr_al_end:
                for param_group in self.log_alpha_optimizer.param_groups:
                    param_group['lr'] = self.lr_al_end
            else:
                self.log_alpha_scheduler.step()


    def reset_alpha(self):
        self.log_alpha = self.init_alpha.detach().clone()
        if self.learn_alpha:
            self.log_alpha.requires_grad = True
            self.log_alpha_optimizer = self.opt_cls([self.log_alpha], lr=self.lr_al, weight_decay=0.01)
        if self.lr_al_schedule:
            self.log_alpha_scheduler = StepLR(
                self.log_alpha_optimizer, step_size=self.lr_al_period, gamma=self.lr_al_decay
        )


    def update(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        log_prob: torch.Tensor,
        update_alpha: bool
    ) -> Tuple[float, float, float]:
        """
        Performs the core policy (actor) update, e.g. SAC-style:
            - Minimizes (or maximizes) Q-values depending on 'actor_type'
            - Updates alpha if needed (automatic temperature tuning)

        Args:
            q1: Q-values from first critic
            q2: Q-values from second critic
            log_prob: log-probability of sampled actions
            update_alpha: whether to update alpha or not (bool)

        Returns:
            (loss_q_eval, loss_entropy, loss_alpha): Float scalars for logging.
        """
        # HINT:
        # 1. Depending on self.actor_type == "min" or "max", combine q1 and q2 (max/min).
        # 2. Compute policy gradient loss with log_prob, alpha, etc.
        # 3. Backprop through self.optimizer.
        # 4. If self.learn_alpha and update_alpha, update log_alpha as well.
        # 5. Return the scalar losses for logging.
        raise NotImplementedError("Actor update logic for Q + alpha * log_prob, etc.")


    def update_policy(self, actor: "Actor"):
        """
        Hard-updates this actor's network weights from another actor's network.
        Typically used in multi-agent or target-network scenarios.
        """
        # HINT:
        # 1. self.net.load_state_dict(actor.net.state_dict())
        # 2. Possibly handle other parameters like alpha, log_alpha, etc.
        raise NotImplementedError("Implement a direct or partial copy of actor.net parameters.")


    def get_action(
        self,
        obsrv: Union[np.ndarray, torch.Tensor],
        agents_action: Optional[Dict[str, np.ndarray]] = None,
        append: Optional[Union[np.ndarray, torch.Tensor]] = None,
        latent: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Returns a deterministic action from the net (self.net.forward),
        possibly using other agents' actions for concatenation.

        Args:
            obsrv: The current observation.
            agents_action: Dict of other agents' actions if multi-agent synergy is needed.
            append: Extra data to append to the observation.
            latent: Latent variables to feed in.
            kwargs: Additional arguments (e.g., time, steps).

        Returns:
            (action, info_dict): The resulting action (always as np array),
            and additional info in a dict (e.g., timing, status).
        """
        # HINT:
        # 1. Possibly combine actions from self.obsrv_list and agents_action => call self.combine_actions(...).
        # 2. With torch.no_grad(), call self.net(...) for a deterministic forward pass.
        # 3. Convert any torch.Tensor output to numpy if needed.
        # 4. Return (action, dict(t_process=..., status=...)).
        raise NotImplementedError("Implement deterministic action retrieval, including multi-agent synergy.")


    def sample(
        self,
        obsrv: Union[np.ndarray, torch.Tensor],
        agents_action: Optional[Dict[str, np.ndarray]] = None,
        append: Optional[Union[np.ndarray, torch.Tensor]] = None,
        latent: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Returns a stochastic action + log_prob if the policy is stochastic.
        Might also incorporate other agents' actions if needed.

        Args:
            obsrv: Current observation (np or torch).
            agents_action: Dictionary of other agents' actions.
            append: Extra data appended to observation.
            latent: Latent variables.
            kwargs: Additional arguments.

        Returns:
            (sampled_action, log_prob): Action (np or torch), log probability (np or torch).
        """
        # HINT:
        # 1. Possibly combine actions if self.obsrv_list is not None.
        # 2. If self.is_stochastic: call self.net.sample(...)
        # 3. Else, raise error or do something else if policy is not stochastic.
        raise NotImplementedError("Implement the sampling logic for a stochastic policy.")





class Critic(BaseBlock):
    """
    Critic block that manages a TwinnedQNetwork (net) and optionally a target network for stable updates.
    Inherits from BaseBlock for config/device integration.
    """

    # Class-level attribute referencing the network:
    net: TwinnedQNetwork

    def __init__(
        self,
        cfg,
        cfg_arch,
        device: torch.device,
        verbose: bool = True
    ) -> None:
        """
        Args:
            cfg: Configuration object (including training/eval flags).
            cfg_arch: Architecture config (e.g., obsrv_dim, mlp_dim, etc.).
            device: torch device ('cpu' or 'cuda').
            verbose: If True, prints debug info.
        """
        super().__init__(cfg, device)

        # Hint:
        # 1. Possibly store mode, update_target_period if not self.eval, etc.
        # 2. Call self.build_network(cfg, cfg_arch, verbose).
        #    That will create self.net and self.target.
        raise NotImplementedError("Fill in Critic.__init__ logic.")


    def build_network(self, cfg, cfg_arch, verbose: bool = True):
        """
        Builds the main TwinnedQNetwork (self.net) plus a target network if training.
        Also loads a pretrained model if cfg_arch specifies a 'pretrained_path'.

        Args:
            cfg: Training config (may specify if eval mode).
            cfg_arch: Arch config with obsrv_dim, mlp_dim, etc.
            verbose: If True, prints architecture info.
        """
        # HINT:
        # 1. Create self.net = TwinnedQNetwork(...).
        # 2. If pretrained_path is given, load model state (but maybe skip some parts).
        # 3. If eval mode: self.net.eval(), freeze params, set self.target = self.net.
        #    else: create a deepcopy => self.target = copy.deepcopy(self.net) => build_optimizer(cfg).
        if not self.eval:
            self.mode: str = cfg.mode
            self.update_target_period = int(cfg.update_target_period)

        self.build_network(cfg, cfg_arch, verbose=verbose)

    def build_network(self, cfg, cfg_arch, verbose: bool = True):

        self.net = TwinnedQNetwork(
            obsrv_dim=cfg_arch.obsrv_dim, mlp_dim=cfg_arch.mlp_dim, action_dim=cfg_arch.action_dim,
            append_dim=cfg_arch.append_dim, latent_dim=cfg_arch.latent_dim, activation_type=cfg_arch.activation,
            device=self.device, verbose=verbose
        )

        # Loads model if specified.
        if hasattr(cfg_arch, "pretrained_path"):
            if cfg_arch.pretrained_path is not None:
                pretrained_path = cfg_arch.pretrained_path
                self.net.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                print(f"--> Loads {self.net_name} from {pretrained_path}.")

        if self.eval:
            self.net.eval()
            for _, param in self.net.named_parameters():
                param.requires_grad = False
            self.target = self.net  # alias
        else:
            self.target = copy.deepcopy(self.net)
            self.build_optimizer(cfg)


    def build_optimizer(self, cfg):
        super().build_optimizer(cfg)
        self.terminal_type: str = cfg.terminal_type
        self.tau = float(cfg.tau)

        # Discount factor
        self.gamma_schedule: bool = cfg.gamma_schedule
        if self.gamma_schedule:
            self.gamma_scheduler = StepLRMargin(
                init_value=cfg.gamma, period=cfg.gamma_period, decay=cfg.gamma_decay, end_value=cfg.gamma_end, goal_value=1.
            )
            self.gamma: float = self.gamma_scheduler.get_variable()
        else:
            self.gamma: float = cfg.gamma

    def update_hyper_param(self) -> bool:
        """Updates the hyper-parameters of the critic, e.g., discount factor.

        Returns:
            bool: True if the discount factor (gamma) is updated.
        """
        if self.eval:
            return False
        else:
        # Uses built scheduler object vs pytorch implmentation
            super().update_hyper_param()
            if self.gamma_schedule:
                old_gamma = self.gamma_scheduler.get_variable()
                self.gamma_scheduler.step()
                self.gamma = self.gamma_scheduler.get_variable()
                if self.gamma != old_gamma:
                    return True
            return False


    def update(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        q1_nxt: torch.Tensor,
        q2_nxt: torch.Tensor,
        non_final_mask: torch.Tensor,
        reward: torch.Tensor,
        g_x: torch.Tensor,
        l_x: torch.Tensor,
        binary_cost: torch.Tensor,
        entropy_motives: torch.Tensor
    ) -> float:
        """
        Performs one gradient update step for the critic using Bellman backup targets.

        Args:
            q1, q2: Current Q estimates from self.net.
            q1_nxt, q2_nxt: Next Q estimates from target (or other) network.
            non_final_mask: Boolean mask indicating valid transitions.
            reward: Immediate reward signal.
            g_x, l_x, binary_cost: Additional terms for specialized cost/reward shaping.
            entropy_motives: Extra term for performance shaping.

        Returns:
            The scalar MSE loss for logging.
        """
        # HINT:
        # 1. Compute y = get_bellman_update(...) or your custom logic.
        # 2. Possibly add gamma * entropy_motives if self.mode == 'performance' and mask is True.
        # 3. Compute MSE loss => (loss_q1 + loss_q2).
        # 4. Zero grad => backprop => optimizer.step().
        # 5. Return loss_q.item().
        raise NotImplementedError("Implement Bellman backup (y) and MSE for Q1/Q2 here.")


    def update_target(self):
        """
        Performs a polyak (soft) update of self.target params from self.net.
        Used after each training step or periodically (self.update_target_period).
        """
        # HINT:
        # e.g. soft_update(self.target, self.net, self.tau)
        raise NotImplementedError("Implement polyak averaging from self.net to self.target.")


    def restore(self, step: int, model_folder: str, verbose: bool = True):
        """
        Loads saved weights from disk for the main (self.net) network.
        If not in eval mode, also loads them for self.target.

        Args:
            step: Checkpoint step/index.
            model_folder: Directory for model checkpoint files.
            verbose: Whether to print debug info.

        Returns:
            The path from which it restored.
        """
        # HINT :
        # 1. Possibly call super().restore(step, model_folder, verbose).
        # 2. Load self.target if not eval mode.
        # 3. Return the path.
        raise NotImplementedError("Implement model loading, including target network if not eval.")

    ####################################################
    # VALUE SIGNATURE
    ####################################################
    def value(
        self,
        obsrv: Union[np.ndarray, torch.Tensor],
        action: Union[np.ndarray, torch.Tensor],
        append: Optional[Union[np.ndarray, torch.Tensor]] = None,
        latent: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> np.ndarray:
        """
        Returns the average Q-value (Q1+Q2)/2 for a given (obsrv, action) pair.

        Args:
            obsrv: Observation input.
            action: Action input.
            append: Additional data appended to obsrv if used in net.
            latent: Latent data appended if used in net.

        Returns:
            Q-value as a numpy array.
        """
        # HINT:
        # 1. with torch.no_grad(): q1, q2 = self.net(obsrv, action, append=append, latent=latent)
        # 2. average = (q1 + q2) / 2
        # 3. convert to numpy if it's a torch.Tensor
        # 4. return average
        raise NotImplementedError("Compute average Q-value from twinned networks in a no_grad context.")


### Key method that allows porting actor and critic into training scripts ###
def build_network(cfg, cfg_arch, device: torch.device,
                  verbose: bool = True) -> Tuple[Dict[str, Critic], Dict[str, Actor]]:
  critics: Dict[str, Critic] = {}
  actors: Dict[str, Actor] = {}
  for idx in range(cfg.num_critics):
    cfg_critic = getattr(cfg, f"critic_{idx}")
    cfg_arch_critic = getattr(cfg_arch, f"critic_{idx}")
    critic = Critic(cfg=cfg_critic, cfg_arch=cfg_arch_critic, verbose=verbose, device=device)
    critics[cfg_critic.net_name] = critic
  assert "central" in critics, "Must have a central critic."