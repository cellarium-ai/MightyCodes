import numpy as np
import torch
from typing import List, Tuple, Union, Optional

from mighty_codes.nn_utils import \
    generate_dense_nnet

class TwoBodyMRFPotential(torch.nn.Module):
    """The base class for the pairwise (two-body) potential between two information
    words.
    
    .. note:: The communication channel, in a more general setting, could may depend
    on the information word. For example, in multiplexed molecular profiling, the number
    of hybridization probes per mRNA is not necessarily uniform across, resulting in a
    net reduction or increase of the readout intensity. We refer to this external
    information as "metadata".
    
    """
    
    def __init__(
            self,
            n_symbols: int):
        super(TwoBodyMRFPotential, self).__init__()
        self.n_symbols = n_symbols
    
    def forward(
            self,
            c_1_bls: torch.Tensor,
            nu_1_bj: Optional[torch.Tensor],
            pi_1_b: torch.Tensor,
            c_2_bls: torch.Tensor,
            nu_2_bj: Optional[torch.Tensor],
            pi_2_b: torch.Tensor) -> torch.Tensor:
        """
        :param c_1_bls: first code with shape B + (code_length, n_symbols)
        :param nu_1_bj: first code metadata with shape B + (n_meta,)
        :param pi_1_b: first code metadata with shape B
        :param c_2_bls: first code with shape B + (code_length, n_symbols)
        :param nu_2_bj: first code metadata with shape B + (n_meta,)
        :param pi_2_b: first code metadata with shape B
        :return: a tensor with shape B
        
        .. note:: B denotes batch shape; it could be empty (), a single
          batch dimension (batch_size,), or more.
        """
        return NotImplementedError
    

class NeuralGeneralPITwoBodyMRFPotential(TwoBodyMRFPotential):
    
    def __init__(
            self,
            n_symbols: int,
            n_meta: int,
            code_length_reduction: torch.nn.Module,
            eta_nnet_specs: List[Union[str, Tuple[int, int]]],
            xi_nnet_specs: List[Union[str, Tuple[int, int]]],
            psi_nnet_specs: List[Union[str, Tuple[int, int]]]):
        
        super(NeuralGeneralPITwoBodyMRFPotential, self).__init__(
            n_symbols=n_symbols)
        self.n_meta = n_meta
        self.code_length_reduction = code_length_reduction
        
        # generate neural nets
        self.eta_nnet = generate_dense_nnet(eta_nnet_specs)
        self.xi_nnet = generate_dense_nnet(xi_nnet_specs)
        self.psi_nnet = generate_dense_nnet(psi_nnet_specs)
    
    def forward(
            self,
            c_1_bls: torch.Tensor,
            nu_1_bj: Optional[torch.Tensor],
            pi_1_b: torch.Tensor,
            c_2_bls: torch.Tensor,
            nu_2_bj: Optional[torch.Tensor],
            pi_2_b: torch.Tensor) -> torch.Tensor:
        
        code_length, n_symbols = c_1_bls.shape[-2:]
        batch_shape = c_1_bls.shape[:-2]
        log_pi_1_b = pi_1_b.log()
        log_pi_2_b = pi_2_b.log()
        
        # generate features
        if self.n_meta == 0:
            
            assert nu_1_bj is None
            assert nu_2_bj is None
            
            rho_1_blf = torch.cat(
                (c_1_bls,
                 log_pi_1_b[..., None, None].expand(batch_shape + (code_length, 1))),
                dim=-1)
            rho_2_blf = torch.cat(
                (c_2_bls,
                 log_pi_2_b[..., None, None].expand(batch_shape + (code_length, 1))),
                dim=-1)
            
        else:
            
            assert nu_1_bj is not None
            assert nu_2_bj is not None
            
            rho_1_blf = torch.cat(
                (c_1_bls,
                 log_pi_1_b[..., None, None].expand(batch_shape + (code_length, 1)),
                 nu_1_bj[..., None, :].expand(batch_shape + (code_length, self.n_meta))),
                dim=-1)
            rho_2_blf = torch.cat(
                (c_2_bls,
                 log_pi_2_b[..., None, None].expand(batch_shape + (code_length, 1)),
                 nu_2_bj[..., None, :].expand(batch_shape + (code_length, self.n_meta))),
                dim=-1)
            
        # process with eta nnet
        eta_bln = self.eta_nnet(rho_1_blf) + self.eta_nnet(rho_2_blf)
        
        # process with xi and reduce over code_lenth dim
        xi_bm = self.code_length_reduction(self.xi_nnet(eta_bln).permute((0, 2, 1)))
        
        # process with psi
        # note: we exponentiate the end result to make it positive (repulsive potential)
        out_b = self.psi_nnet(xi_bm).exp().squeeze(-1)
        
        return out_b


class TwoBodyPotentialPrefactor(torch.nn.Module):
    """The base class for two-body potential prefactor providers.
    
    .. note:: A class of variational two-body potentials could be thought of
      as being factorizable as follows:
    
          S_{2B}(c_1, pi_1, nu_1; c_2, pi_2, nu_2) =
              f^1(pi_1, nu_1; pi_2, nu_2) S^1(c_1, c_2) +
              f^2(pi_1, nu_1; pi_2, nu_2) S^2(c_1, c_2) +
              f^3(pi_1, nu_1; pi_2, nu_2) S^3(c_1, c_2) + ...,
            
      where we have factored out the dependence on pi and nu as a separate
      symmetric function. The series is naturally truncated at finite order.
      This class is the base class of all concrete implementation of the
      "f" part in the expansion.
    """
    
    def __init__(self, n_meta: int):
        super(TwoBodyPotentialPrefactor, self).__init__()
        self.n_meta = n_meta
    
    def forward(
            self,
            nu_1_bj: Optional[torch.Tensor],
            pi_1_b: torch.Tensor,
            nu_2_bj: Optional[torch.Tensor],
            pi_2_b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    
class BiLinearTwoBodyPotentialPrefactor(TwoBodyPotentialPrefactor):
    """A concrete implementation of `TwoBodyPotentialPrefactor` as
    follows:
    
        f(pi_1, nu_1; pi_2, nu_2) =  c_0 +  c_1 * (pi_1 + pi_2) + c_2 * pi_1 * pi_2,
            
    where c_0, c_1, and c_2 are learnable positive parametrs.
    
    .. note:: the metadata variables are ignored (not implemeneted).
    """
    
    def __init__(
            self,
            n_components: int,
            init_bilinear_scale: float,
            init_linear_scale: float,
            init_constant_scale: float):
        """
        
        :param n_components: number of expansion components (refer to the
          docstring of `TwoBodyPotentialPrefactor`)
        :param init_bilinear_scale: initial bilinear term
        :param init_linear_scale: initial linear term
        :param init_constant_scale: initial constant term
        """
        super(BiLinearTwoBodyPotentialPrefactor, self).__init__(n_meta=0)
        self.n_components = n_components
    
        # bilinear scale
        self.beta_bilinear_m_unconstrained = torch.nn.Parameter(
            np.log(init_bilinear_scale) + torch.rand(n_components).log())

        # linear parameters
        self.beta_linear_m_unconstrained = torch.nn.Parameter(
            np.log(init_linear_scale) + torch.rand(n_components).log())
        
        # constant parameters
        self.beta_constant_m_unconstrained = torch.nn.Parameter(
            np.log(init_constant_scale) + torch.rand(n_components).log())

    @property
    def beta_bilinear_m(self):
        return self.beta_bilinear_m_unconstrained.exp()
    
    @property
    def beta_linear_m(self):
        return self.beta_linear_m_unconstrained.exp()
    
    @property
    def beta_constant_m(self):
        return self.beta_constant_m_unconstrained.exp()

    def forward(
            self,
            nu_1_bj: Optional[torch.Tensor],
            pi_1_b: torch.Tensor,
            nu_2_bj: Optional[torch.Tensor],
            pi_2_b: torch.Tensor) -> torch.Tensor:
        
        assert nu_1_bj is None
        assert nu_2_bj is None
        
        batch_shape = pi_1_b.shape
        pi_1_bm = pi_1_b[..., None].expand(batch_shape + (self.n_components,))
        pi_2_bm = pi_2_b[..., None].expand(batch_shape + (self.n_components,))
        prefactor_bm = (
            self.beta_constant_m +
            self.beta_linear_m * (pi_1_bm + pi_2_bm) +
            self.beta_bilinear_m * pi_1_bm * pi_2_bm)
        
        return prefactor_bm

    
class BiLinearTwoBodyPotentialPrefactorSimple(TwoBodyPotentialPrefactor):
    """A concrete implementation of `TwoBodyPotentialPrefactor` as
    follows:
    
        f(pi_1, nu_1; pi_2, nu_2) =  pi_1 * pi_2
            
    .. note:: there are no learnable parameters.
    .. note:: the metadata variables are ignored (not implemeneted).
    """
    
    def __init__(
            self,
            n_components: int):
        super(BiLinearTwoBodyPotentialPrefactorSimple, self).__init__(n_meta=0)
        self.n_components = n_components
    
    def forward(
            self,
            nu_1_bj: Optional[torch.Tensor],
            pi_1_b: torch.Tensor,
            nu_2_bj: Optional[torch.Tensor],
            pi_2_b: torch.Tensor) -> torch.Tensor:
        
        assert nu_1_bj is None
        assert nu_2_bj is None

        batch_shape = pi_1_b.shape
        pi_1_bm = pi_1_b[..., None].expand(batch_shape + (self.n_components,))
        pi_2_bm = pi_2_b[..., None].expand(batch_shape + (self.n_components,))
        prefactor_bm = pi_1_bm * pi_2_bm
        
        return prefactor_bm


class NeuralTwoBodyPotentialPrefactor(TwoBodyPotentialPrefactor):
    """An implementation of `TwoBodyPotentialPrefactor` via generic
    neural networks.
    
    .. note:: the 1 <=> 2 symmetry is strictly enforced in the
      architecture.
      
    .. note:: the metadata variables are ignored (not implemeneted).
    """
    
    def __init__(
            self,
            n_components: int,
            eta_nnet_specs: List[Union[str, Tuple[int, int]]],
            psi_nnet_specs: List[Union[str, Tuple[int, int]]]): 
        super(NeuralTwoBodyPotentialPrefactor, self).__init__(n_meta=0)
        self.n_components = n_components
        
        # generate neural nets
        self.eta_nnet = generate_dense_nnet(eta_nnet_specs)
        self.psi_nnet = generate_dense_nnet(psi_nnet_specs)
        
    def forward(
            self,
            nu_1_bj: Optional[torch.Tensor],
            pi_1_b: torch.Tensor,
            nu_2_bj: Optional[torch.Tensor],
            pi_2_b: torch.Tensor) -> torch.Tensor:
        
        assert nu_1_bj is None
        assert nu_2_bj is None
        
        log_pi_1_b1 = pi_1_b.log().unsqueeze(-1)
        log_pi_2_b1 = pi_2_b.log().unsqueeze(-1)
        
        # process with eta nnet
        out_bm = self.psi_nnet(
            self.eta_nnet(log_pi_1_b1) +
            self.eta_nnet(log_pi_2_b1)).exp()
        
        assert out_bm.shape == pi_1_b.shape + (self.n_components,)
        
        return out_bm

    
class ExpBiLinearPITwoBodyMRFPotential(TwoBodyMRFPotential):
    """A concrete implementation of `TwoBodyMRFPotential` as
    follows:
    
          S_{2B}(c_1, pi_1, nu_1; c_2, pi_2, nu_2) = 
              (1/N) \sum_{j=1}^N [f^j(pi_1, nu_1; pi_2, nu_2) S^j(c_1, c_2)]
              
    where the series is truncated at `n_components`. 

    The S-functions are modeled as learnable bilinears forms. The
    f-functions must be provided as some implementation of
    `TwoBodyPotentialPrefactor`.    
    """
    
    def __init__(
            self,
            n_symbols: int,
            n_components: int,
            code_length_reduction_type: str,
            prefactor_provider: TwoBodyPotentialPrefactor,
            init_bilinear_diag_scale: float,
            init_bilinear_rand_scale: float,
            init_linear_scale: float,
            init_constant_scale: float):
        super(ExpBiLinearPITwoBodyMRFPotential, self).__init__(n_symbols=n_symbols)

        self.n_components = n_components
        self.prefactor_provider = prefactor_provider
        self.code_length_reduction_type = code_length_reduction_type
        
        assert code_length_reduction_type in {'mean', 'sum'}
        
        # S-function bilinear parameters
        self.gamma_mss_unconstrained = torch.nn.Parameter(
            init_bilinear_diag_scale * torch.eye(n_symbols)[None, ...].expand(
                (n_components, n_symbols, n_symbols)).contiguous() +
            init_bilinear_rand_scale * torch.randn(
                (n_components, n_symbols, n_symbols)))
        
        # S-function linear parameters
        self.gamma_ms = torch.nn.Parameter(
            init_linear_scale * torch.randn((n_components, n_symbols)))
        
        # S-function constant parameters
        self.gamma_m = torch.nn.Parameter(
            init_constant_scale * torch.randn((n_components,)))
        
    @property
    def gamma_mss(self):
        return 0.5 * (self.gamma_mss_unconstrained + self.gamma_mss_unconstrained.permute(0, 2, 1))
    
    def forward(
            self,
            c_1_bls: torch.Tensor,
            nu_1_bj: Optional[torch.Tensor],
            pi_1_b: torch.Tensor,
            c_2_bls: torch.Tensor,
            nu_2_bj: Optional[torch.Tensor],
            pi_2_b: torch.Tensor) -> torch.Tensor:
        
        code_length, n_symbols = c_1_bls.shape[-2:]
        batch_shape = c_1_bls.shape[:-2]
        if self.code_length_reduction_type == 'mean':
            code_length_norm_factor = 1. / code_length
        elif self.code_length_reduction_type == 'sum':
            code_length_norm_factor = 1.
        else:
            raise ValueError('Bad code_length_reduction_type;  allowed values are: mean, sum')
        
        # code-code interaction term
        # note: no code-length normalization
        c_exp_bilinear_bm = (
            code_length_norm_factor * torch.einsum(
                '...ls,...lr,msr->...m',
                c_1_bls,
                c_2_bls,
                self.gamma_mss) +
            code_length_norm_factor * torch.einsum(
                '...ls,ms->...m',
                c_1_bls + c_2_bls,
                self.gamma_ms)
            + self.gamma_m).exp()
            
        # calculate prefactor
        prefactors_bm = self.prefactor_provider(
            nu_1_bj=nu_1_bj,
            pi_1_b=pi_1_b,
            nu_2_bj=nu_2_bj,
            pi_2_b=pi_2_b)
        
        # we avergae over components
        out_b = (c_exp_bilinear_bm * prefactors_bm).mean(-1)
        
        return out_b
