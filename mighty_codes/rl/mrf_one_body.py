import numpy as np
import torch
from typing import List, Tuple, Union, Dict

from mighty_codes.nn_utils import \
    generate_dense_nnet, \
    assert_nn_specs_io_features


class OneBodyMRFPotential(torch.nn.Module):
    def __init__(
            self,
            n_symbols: int):
        super(OneBodyMRFPotential, self).__init__()
        self.n_symbols = n_symbols
    
    def forward(
            self,
            c_bls: torch.Tensor,
            n_types: torch.Tensor,
            type_rank: torch.Tensor,
            pi: torch.Tensor,
            pi_cdf: torch.Tensor):
        """
        :paraam c_bls: code with shape (batch_size, code_length, n_symbols)
        :param n_types: number of types (scalar)
        :param type_rank: rank of the code (scalar)
        :param pi: source prior PMF (scalar)
        :param pi_cdf: source prior CDF (scalar)
        """
        raise NotImplementedError


class NeuralPIOneBodyMRFPotential(OneBodyMRFPotential):
    def __init__(
            self,
            n_symbols: int,
            nn_specs: List[Union[str, Tuple[int, int]]],
            lp_norm: int,
            assert_shapes: bool = True):
        super(NeuralPIOneBodyMRFPotential, self).__init__(n_symbols=n_symbols)
        
        self.assert_shapes = assert_shapes
        
        # we need 6 input features (see below)
        # we need n_symbols + 1 output features: symbol weights + strength
        assert_nn_specs_io_features(nn_specs, in_features=6, out_features=(n_symbols + 1))
        self.nn = generate_dense_nnet(nn_specs)
        self.lp_norm = lp_norm
    
    def get_one_body_potential_props(
            self,
            n_types_b: torch.Tensor,
            code_length_b: torch.Tensor,
            type_rank_b: torch.Tensor,
            pi_b: torch.Tensor,
            pi_cdf_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        if self.assert_shapes:
            shape = n_types_b.shape
            assert code_length_b.shape == shape
            assert type_rank_b.shape == shape
            assert pi_b.shape == shape
            assert pi_cdf_b.shape == shape
        
        x_bf = torch.cat((
            torch.log(n_types_b).unsqueeze(-1),
            np.log(self.n_symbols) * torch.ones_like(n_types_b).unsqueeze(-1),
            torch.log(code_length_b).unsqueeze(-1),
            type_rank_b.unsqueeze(-1),
            pi_b.unsqueeze(-1),
            pi_cdf_b.unsqueeze(-1)), dim=-1)
        
        # get nnet output
        out_bo = self.nn(x_bf)
        
        # calculate props
        symbol_weights_bs = torch.softmax(out_bo[..., :self.n_symbols], -1)
        potential_strength_b = out_bo[..., self.n_symbols].exp()
        
        return {
            'symbol_weights_bs': symbol_weights_bs,
            'potential_strength_b': potential_strength_b
        }
        
    def forward(
            self,
            c_bls: torch.Tensor,
            n_types: torch.Tensor,
            type_rank: torch.Tensor,
            pi: torch.Tensor,
            pi_cdf: torch.Tensor):
   
        # add a dummy batch dimension and get one-body potential props
        code_length = c_bls.shape[-2]
        props_dict = self.get_one_body_potential_props(
            n_types_b=n_types.unsqueeze(-1),
            code_length_b=code_length * torch.ones_like(n_types).unsqueeze(-1),
            type_rank_b=type_rank.unsqueeze(-1),
            pi_b=pi.unsqueeze(-1),
            pi_cdf_b=pi_cdf.unsqueeze(-1))
        
        # calculate potential
        target_symbol_weights_s = props_dict['symbol_weights_bs'][0, :]
        potential_strength = props_dict['potential_strength_b'][0]
        code_symbol_weights_bs = c_bls.mean(-2)
        potential_b = potential_strength * (code_symbol_weights_bs - target_symbol_weights_s) \
            .abs() \
            .pow(self.lp_norm) \
            .sum(-1)
        
        return potential_b

