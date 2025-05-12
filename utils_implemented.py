import torch
import numpy as np
from typing import Optional, Union, List, Dict

"""
Note that to facilitate smooth pipeline with other aspects of project in terms of code. A member of our three person team
observed header information from methods in repository, provided header descriptions, and the other two members were responsible for 
filling out the core code for implemented code. (Similar to code assignment format)
"""



####################################################################################
# soft_update (Self-Implemented) based of Homework 6 Actor-Critic
#####################################################################################
def soft_update(
    target: torch.nn.Module,
    source: torch.nn.Module,
    tau: float
) -> None:
    """
    Performs a soft (Polyak) update of target network parameters
    toward source network parameters.

    Args:
        target: the target network (parameters updated in place)
        source: the source network from which parameters are sampled
        tau: blend factor in [0,1]. 0 => no update, 1 => full copy.
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data = (1 - tau)*target_param.data + tau*param.data
