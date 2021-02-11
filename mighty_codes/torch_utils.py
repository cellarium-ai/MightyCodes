import torch
import numpy as np
import math

from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Dict, Any, Optional, Union, Generator


def to_np(input: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts a torch tensor to ndarray. If the input is ndarray, it is returned
    unchanhed."""
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        return np.asarray(input)

    
def to_torch(
        input: Union[torch.Tensor, np.ndarray],
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
    """Converts a number or ndarray to a torch tensor. If the input is a torch tensor,
    it is returned unchanged."""
    if isinstance(input, torch.Tensor):
        return input.type(dtype).to(device)
    elif isinstance(input, np.ndarray):
        return torch.tensor(input, device=device, dtype=dtype)
    else:
        return torch.tensor(np.asarray(input), device=device, dtype=dtype)

    
def to_simplex(input: torch.Tensor) -> torch.Tensor:
    """Puts the tensor on simplex by applying a softmax along the last dimension."""
    return torch.nn.functional.softmax(input, dim=-1)


def to_one_hot_encoded(
        type_indices_n: torch.LongTensor,
        n_types: int) -> torch.Tensor:
    """Takes an index tensor and yields a one-hot encoded representation"""
    identity_tt = torch.eye(n=n_types, device=type_indices_n.device, dtype=torch.long)
    return identity_tt[type_indices_n, :]


def split_tensors(
        tensors: Tuple[torch.Tensor],
        dim: int,
        split_size: int,
        validate_input: bool = True) -> Generator[Tuple[torch.Tensor], None, None]:
    size = tensors[0].size(dim)
    n_splits = math.ceil(size / split_size)
    if validate_input:
        assert dim >= 0
        assert split_size >= 1
        assert len(tensors) > 0
        for t in tensors:
            assert t.ndim >= dim
            assert t.size(dim) == size
    joint_split = tuple(torch.split(t, split_size, dim) for t in tensors)
    for i_split in range(n_splits):
        yield tuple(s[i_split] for s in joint_split)
