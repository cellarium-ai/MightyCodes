import numpy as np
import torch
from torch.distributions.utils import broadcast_all
from typing import List

# ln(1/2)
LN_HALF = -0.6931471805599453


def log_prob_complement(log_prob: torch.Tensor) -> torch.Tensor:
    """Calculates the complement of a probability in the natural log scale,
    log(1 - exp(log_prob)), in a numerically stable fashion.
    """
    log_prob_zero_capped = torch.clamp(log_prob, max=0.)
    return torch.where(
        log_prob_zero_capped >= LN_HALF,
        torch.log(-torch.expm1(log_prob_zero_capped)),
        torch.log1p(-torch.exp(log_prob_zero_capped)))

    
def int_to_base(n: int, b: int) -> List[int]:
    """Converts an integer to the binary representation and returns the result
    as a list of 0s and 1s."""
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def left_zero_pad(l: List[int], n: int) -> List[int]:
    """Takes a list of integers and zero pads it from the left with [0]s."""
    return [0] * (n - len(l)) + l


def get_exp_decay(init_value: float, final_value: float, n_iters_decay: int, i_iter: int) -> float:
    beta_decay = np.log(init_value / final_value) / n_iters_decay
    return init_value * np.exp(-beta_decay * i_iter)


def log_binom(k, n):
    return torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)


def logaddexp(a: torch.Tensor, b: torch.Tensor):
    a, b = broadcast_all(a, b)
    return torch.stack((a, b), -1).logsumexp(-1)

