import sys
import numpy as np
import torch

from typing import List, Dict, Optional, NamedTuple

from mighty_codes.channels import \
    SingleEntityChannelModel, \
    BinaryChannelSpecification, \
    BinaryAsymmetricChannelModel, \
    GaussianChannelModel

from mighty_codes.torch_utils import to_torch

import mighty_codes.consts as consts


class ExperimentSpecification(NamedTuple):
    name: str
    n_types: int
    n_symbols: int
    code_length: int
    min_symbol_weight_s: List[int]
    max_symbol_weight_s: List[int]
    pi_t: np.ndarray


def int_list_repr(l: List[int]) -> str:
    return '(' + ','.join(list(map(str, l))) + ')'


def generate_experiment_spec(
        name_prefix: str,
        min_symbol_weight_s: List[int],
        max_symbol_weight_s: List[int],
        n_symbols: int = 2,
        code_length: Optional[int] = None,
        n_types: Optional[int] = None,
        log_p: Optional[float] = None,
        source_nonuniformity: Optional[float] = None,
        min_code_length: int = 4,
        max_code_length: int = 10,
        min_source_nonuniformity: float = 1e1,
        max_source_nonuniformity: float = 1e3,
        min_n_types: int = 2,
        max_n_types: int = 200,
        min_packing_ratio: float = 1e-3,
        max_packing_ratio: float = 0.75) -> ExperimentSpecification:
    """A helper function for generating a specific coding problem, or sampling one
    from a range of possible problems."""
    
    assert len(min_symbol_weight_s) == n_symbols
    assert len(max_symbol_weight_s) == n_symbols
    
    # choose code length
    if code_length is None:
        code_length = np.random.randint(min_code_length, max_code_length + 1)
        
    # choose n_types
    if n_types is None:
        code_space_size = n_symbols ** code_length
        n_types_lo = max(min_n_types, int(min_packing_ratio * code_space_size))
        n_types_hi = max(n_types_lo, min(max_n_types, int(max_packing_ratio * code_space_size)))
        n_types = np.random.randint(n_types_lo, n_types_hi + 1)
    
    # choose prior
    if log_p is None:
        if source_nonuniformity is None:
            log_p_hi = - np.log(min_source_nonuniformity) / (n_types - 1)
            log_p_lo = - np.log(max_source_nonuniformity) / (n_types - 1)
            log_p = np.random.rand() * (log_p_hi - log_p_lo) + log_p_lo
        else:
            log_p = - np.log(source_nonuniformity) / (n_types - 1)
    
    pi_t = np.exp(log_p) ** np.arange(n_types).astype(np.float)
    pi_t = pi_t / np.sum(pi_t)
    
    if source_nonuniformity is None:
        source_nonuniformity = pi_t[0] / pi_t[-1]
        
    return ExperimentSpecification(
        name=(f"{name_prefix}__"
              f"s_{n_symbols}__"
              f"l_{code_length}__"
              f"t_{n_types}__"
              f"nu_{int(source_nonuniformity)}__"
              f"minsw_{int_list_repr(min_symbol_weight_s)}__"
              f"maxsw_{int_list_repr(max_symbol_weight_s)}"),
        n_symbols=n_symbols,
        code_length=code_length,
        min_symbol_weight_s=min_symbol_weight_s,
        max_symbol_weight_s=max_symbol_weight_s,
        n_types=n_types,
        pi_t=pi_t)


class ChannelModelSpecification(NamedTuple):
    name: str
    channel_model: SingleEntityChannelModel


class SingleEntityCodingProblemSpecification(NamedTuple):
    experiment_spec: ExperimentSpecification
    channel_spec: ChannelModelSpecification
        
    @property
    def name(self) -> str:
        return self.experiment_spec.name + "__" + self.channel_spec.name

# BSC channel p_01 = p_10 = 0.10
channel_bsc_10 = ChannelModelSpecification(
    name="bsc_10",
    channel_model=BinaryAsymmetricChannelModel(
        channel_spec_list=[BinaryChannelSpecification(p_01=0.10, p_10=0.10)],
        device=consts.DEFAULT_DEVICE_EVAL,
        dtype=consts.DEFAULT_DTYPE))

# MERFISH arbitrary-bit BAC channel
channel_bac_merfish = ChannelModelSpecification(
    name="bac_merfish",
    channel_model=BinaryAsymmetricChannelModel(
        channel_spec_list=[BinaryChannelSpecification(p_01=0.04, p_10=0.10)],
        device=consts.DEFAULT_DEVICE_EVAL,
        dtype=consts.DEFAULT_DTYPE))

# MERFISH Gaussian channel 
channel_gaussian_merfish = ChannelModelSpecification(
    name="gaussian_merfish",
    channel_model=GaussianChannelModel(
        n_symbols=2,
        n_readout=1,
        loc_sr=to_torch(
            np.asarray([[0.], [4.36064674]]),
            device=consts.DEFAULT_DEVICE_EVAL,
            dtype=consts.DEFAULT_DTYPE),
        scale_sr=to_torch(
            np.asarray([[1.], [2.03656312]]),
            device=consts.DEFAULT_DEVICE_EVAL,
            dtype=consts.DEFAULT_DTYPE),
        device=consts.DEFAULT_DEVICE_EVAL,
        dtype=consts.DEFAULT_DTYPE))
