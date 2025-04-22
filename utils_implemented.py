import torch
import numpy as np
from typing import Optional, Union, List, Dict


def get_bellman_update(
    mode: str, batch_size: int, q1_nxt: torch.Tensor, q2_nxt: torch.Tensor, non_final_mask: torch.Tensor,
    reward: torch.Tensor, g_x: torch.Tensor, l_x: torch.Tensor, binary_cost: torch.Tensor, gamma: float,
    terminal_type: Optional[str] = None
):

    target_q = torch.min(q1_nxt, q2_nxt).view(-1)
    y = torch.zeros(batch_size).float().to(q1_nxt) 
    final_mask = ~non_final_mask
    terminal_target = torch.min(l_x[non_final_mask], g_x[non_final_mask])
    original_target = torch.min(g_x[non_final_mask], torch.max(l_x[non_final_mask], target_q))
    y[non_final_mask] = (1.0-gamma) * terminal_target + gamma*original_target
    y[final_mask] = torch.min(l_x[final_mask], g_x[final_mask])

    return y


def soft_update(
    target: torch.nn.Module,
    source: torch.nn.Module,
    tau: float
) -> None:
    """
    Performs a soft (Polyak) update of target network parameters
    toward source network parameters.

    Args:
        target: The target network (parameters updated in-place).
        source: The source network from which parameters are sampled.
        tau: Blend factor in [0,1]. 0 => no update, 1 => full copy.
             
    """
    # HINT:
    # 1) Loop over pairs of parameters: (target_param, param) in zip(target.parameters(), source.parameters()).
    # 2) Update each: target_param.data = (1 - tau)*target_param.data + tau*param.data

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data = (1 - tau)*target_param.data + tau*param.data
