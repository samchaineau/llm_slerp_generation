import torch

def slerp(t: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, DOT_THRESHOLD: float = 0.9995, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two vectors.

    Args:
        t (torch.Tensor): Interpolation parameter. Float tensor between 0.0 and 1.0.
        v0 (torch.Tensor): Starting vector.
        v1 (torch.Tensor): Final vector.
        DOT_THRESHOLD (float): Threshold for considering the two vectors as colinear. Default is 0.9995.
        eps (float): Small value to avoid division by zero. Default is 1e-8.

    Returns:
        torch.Tensor: Interpolated vector between v0 and v1.
    """
    # Copy the vectors to reuse them later
    v0_copy = v0.clone().detach()
    v1_copy = v1.clone().detach()

    # Normalize the vectors to get the directions and angles
    v0 = normalize_torch(v0, eps)
    v1 = normalize_torch(v1, eps)

    # Dot product with the normalized vectors
    dot = torch.sum(v0 * v1, dim=-1)

    return slerp_torch(dot, t, v0_copy, v1_copy)

def lerp_torch(t: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Linearly interpolate between two vectors (optimized for torch.Tensor).

    Args:
        t (torch.Tensor): Interpolation parameter. Float tensor between 0.0 and 1.0.
        v0 (torch.Tensor): Starting vector.
        v1 (torch.Tensor): Final vector.

    Returns:
        torch.Tensor: Interpolated vector between v0 and v1.
    """
    return (1 - t).unsqueeze(-1).expand_as(v0) * v0 + t.unsqueeze(-1).expand_as(v1) * v1

def slerp_torch(dot: torch.Tensor, t: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between two vectors (optimized for torch.Tensor).

    Args:
        dot (torch.Tensor): Dot product of the two vectors.
        t (torch.Tensor): Interpolation parameter. Float tensor between 0.0 and 1.0.
        v0 (torch.Tensor): Starting vector.
        v1 (torch.Tensor): Final vector.

    Returns:
        torch.Tensor: Interpolated vector between v0 and v1.
    """
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)

    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)

    # Finish the slerp algorithm
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    res = (s0.unsqueeze(-1).expand_as(v0) * v0) + (s1.unsqueeze(-1).expand_as(v1) * v1)

    return res

def normalize_torch(v: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Normalize a vector (optimized for torch.Tensor).

    Args:
        v (torch.Tensor): Input vector.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Normalized vector.
    """
    norm_v = torch.norm(v)
    if norm_v > eps:
        v = v / norm_v
    return v

def get_odd_sl(tensor: torch.Tensor):
    """
    Get an odd sequence length.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with odd sequence length.
        str: Information on the action taken.
    """
    bs, sl, dim = tensor.shape
    if sl <= 3:
        return tensor, "keep"
    elif sl % 2 != 0:
        to_append = torch.zeros((bs, 1, dim), device=tensor.device)
        last_tok = tensor[:, -1:, :]
        return torch.cat((to_append, tensor[:, :-1, :], to_append, tensor[:, -1:, :], to_append), dim=1), "even"
    else:
        to_append = torch.zeros((bs, 1, dim), device=tensor.device)
        return torch.cat((to_append, tensor, to_append), dim=1), "odd"

def preslice(tensor: torch.Tensor):
    """
    Slice the tensor to form pairs of elements.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Sliced tensor.
    """
    bs, sl, dim = tensor.shape
    return tensor.reshape(bs, int(sl / 2), 2, dim)

def reformat_sequence(tensor: torch.Tensor):
    """
    Reformat the sequence length.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Reformatted tensor.
        torch.Tensor: Temporary tensor for interpolation.
    """
    reformatted, to_do = get_odd_sl(tensor)

    if to_do == "keep":
        return reformatted, None
    else:
        sliced = preslice(reformatted)
        bs, sl, _, dim = sliced.shape
        if to_do == "odd":
            temps = torch.full((bs, int(sl) - 2),
                               0.5,
                               device=tensor.device)
            temps = torch.cat((torch.tensor([[1.0]], device=tensor.device),
                               temps,
                               torch.tensor([[0.0]], device=tensor.device)), dim=1)
        else:
            temps = torch.full((bs, int(sl) - 3),
                               0.5,
                               device=tensor.device)
            temps = torch.cat((torch.tensor([[1.0]], device=tensor.device),
                               temps,
                               torch.tensor([[0.0]], device=tensor.device),
                               torch.tensor([[0.0]], device=tensor.device)), dim=1)
        return sliced, temps

def merge_tokens(tensor: torch.Tensor):
    """
    Merge tokens using spherical linear interpolation.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Merged tensor.
    """
    sliced, temps = reformat_sequence(tensor)
    if temps is None:
        return sliced
    else:
        return slerp(temps, sliced[:, :, 0, :], sliced[:, :, 1, :])
