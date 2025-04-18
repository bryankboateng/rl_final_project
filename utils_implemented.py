import torch


def get_bellman_update(
    mode: str,
    batch_size: int,
    q1_nxt: torch.Tensor,
    q2_nxt: torch.Tensor,
    non_final_mask: torch.Tensor,
    reward: torch.Tensor,
    g_x: torch.Tensor,
    l_x: torch.Tensor,
    binary_cost: torch.Tensor,
    gamma: float,
    terminal_type: str = None
) -> torch.Tensor:
    """
    Computes the Bellman backup (target) values for various RL modes
    (e.g., 'reach-avoid', 'safety', 'performance', 'risk').

    Args:
        mode: The RL mode or objective ('risk', 'reach-avoid', 'safety', 'performance').
        batch_size: Number of transitions in the batch.
        q1_nxt: Q-values from the next state for the first Q-network.
        q2_nxt: Q-values from the next state for the second Q-network.
        non_final_mask: Boolean mask indicating which transitions are not terminal.
        reward: Immediate reward tensor (for 'performance' mode).
        g_x: Some cost or constraint function (e.g., "goal" function).
        l_x: Another cost or constraint function (e.g., "loss" function).
        binary_cost: Typically a {0,1} cost for 'risk' or classification-based tasks.
        gamma: Discount factor (0 < gamma <= 1).
        terminal_type: Optional string specifying how to handle terminal states ('g', 'all', etc.).

    Returns:
        A torch.Tensor of shape [batch_size], representing the Bellman target values.
    """

    # HINT:
    # 1) Depending on mode, define how to combine q1_nxt and q2_nxt (e.g., min or max).
    # 2) Build target_q.
    # 3) Initialize y = zeros(batch_size). 
    # 4) Based on mode logic, fill y for non_final_mask and final_mask.
    # 5) Return y.

    # Initialize target values
    y = torch.zeros(batch_size)

    if (mode == 'risk'):
        next_qs = torch.min(q1_nxt, q2_nxt)
        # use Bellman equation for non-terminal states
        y[non_final_mask] = reward[non_final_mask] + gamma * next_qs[non_final_mask]
        # Use binary cost for terminal states
        y[~non_final_mask] = binary_cost[~non_final_mask]

    elif (mode == 'reach-avoid'):
        next_qs = torch.min(q1_nxt, q2_nxt)
        next_qs = torch.min(g_x, torch.max(l_x, next_qs))
        # use Bellman equation for non-terminal states
        y[non_final_mask] = reward[non_final_mask] + gamma * next_qs[non_final_mask]
        # Use g_x for terminal states
        y[~non_final_mask] = g_x[~non_final_mask]

    elif (mode == 'safety'):
        next_qs = torch.min(q1_nxt, q2_nxt)
        next_qs = torch.min(g_x, next_qs)
        # use Bellman equation for non-terminal states
        y[non_final_mask] = reward[non_final_mask] + gamma * next_qs[non_final_mask]
        # Use g_x for terminal states
        y[~non_final_mask] = g_x[~non_final_mask]

    elif (mode == 'performance'):
        next_qs = torch.max(q1_nxt, q2_nxt)
        # Non-terminal states
        y[non_final_mask] = reward[non_final_mask] + gamma * next_qs[non_final_mask]
        # Terminal states
        y[~non_final_mask] = reward[~non_final_mask]

    else:
        raise ValueError("Mode not supported")
    
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
