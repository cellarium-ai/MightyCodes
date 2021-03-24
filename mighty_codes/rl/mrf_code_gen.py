import numpy as np
import torch
from boltons.cacheutils import cachedproperty, cachedmethod
from typing import Dict, Any, Callable, Optional, Dict, List, Set, Tuple, Union

from mighty_codes.torch_utils import \
    to_torch

from mighty_codes.math_utils import \
    int_to_base, \
    left_zero_pad

from mighty_codes.rl.mrf_one_body import \
    OneBodyMRFPotential

from mighty_codes.rl.mrf_two_body import \
    TwoBodyMRFPotential


class MRFCodebookGenerator(torch.nn.Module):
    
    def __init__(
            self,
            one_body_potential: OneBodyMRFPotential,
            two_body_potential: TwoBodyMRFPotential,
            n_symbols: int,
            code_space_filter_func: Callable,
            device: torch.device,
            dtype: torch.dtype,
            assert_shapes: bool = True):
        """An auto-regressive codebook generator."""
        
        super(MRFCodebookGenerator, self).__init__()
                
        self.one_body_potential = one_body_potential
        self.two_body_potential = two_body_potential
        self.n_symbols = n_symbols
        
        self.device = device
        self.dtype = dtype
        self.assert_shapes = assert_shapes
            
        # code filter
        self.code_space_filter_func = code_space_filter_func
        
        # cache
        self._indexed_code_space_cache = dict()
        
    @cachedmethod(cache='_indexed_code_space_cache')
    def get_code_space_size(
            self,
            code_length: int):
        return self.get_indexed_code_space(code_length).shape[0]

    @cachedmethod(cache='_indexed_code_space_cache')
    def get_indexed_code_space(
            self,
            code_length: int):
        n_all_codes = self.n_symbols ** code_length
        all_codes_al_np = np.zeros((n_all_codes, code_length), dtype=np.int8)
        for i_code in range(n_all_codes):
            all_codes_al_np[i_code, :] = left_zero_pad(int_to_base(i_code, self.n_symbols), code_length)
        all_codes_als_np = np.eye(self.n_symbols, dtype=np.int8)[all_codes_al_np, :]
        passing_a = self.code_space_filter_func(all_codes_als_np, code_length)
        return all_codes_als_np[passing_a, ...]
    
    @cachedmethod(cache='_indexed_code_space_cache')
    def get_indexed_code_space_symbol_count(
            self,
            code_length: int):
        all_codes_als = self.get_indexed_code_space(code_length)
        return all_codes_als.sum(axis=-2)

    def forward(
            self,
            code_length: int,
            n_types: int,
            batch_size: int,
            pi_t: Union[torch.Tensor, np.ndarray],
            nu_tj: Optional[torch.Tensor],
            two_body_max_n_interactions: int,
            two_body_dropout_policy: str,
            top_proposals_per_column: int,
            random_proposals_per_column: int,
            disable_one_body: bool = False,
            disable_two_body: bool = False,
            enable_hardcore_two_body_potential: bool = False,
            action_policy: str = 'sampled',
            action_policy_epsilon: float = 0.2,
            one_body_strength_multiplier: float = 1.0,
            two_body_strength_multiplier: float = 1.0,
            top_k_noise_std: float = 0.):
        """
        :param code_length: code length (codebook rows)
        :param n_types: number of words (codebook columns)
        :param batch_size: codebook batch size
        :param pi_t: source distribution
        :param nu_tj: source metadata
        :param two_body_max_n_interactions: max number of two-body interaction terms to keep
        :param two_body_dropout_policy: policy to drop extra interaction terms
        :param top_proposals_per_column: number of top proposals kept for each codebook column
        :param random_proposals_per_column: number of random proposals for each codebook column
        :param disable_one_body: disable one-body potential
        :param disable_two_body: disable two-body potential
        :param enable_hardcore_two_body_potential: explicitly include a hardcore two-body interaction term
        :param action_policy: how to choose action? choices: 'sampled', 'greedy', 'epsilon-greedy'
        :param action_policy_epsilon: epislon for epsilon-greedy action policy
        :param one_body_strength_multiplier: scale-factor for the one-body potential term
        :param two_body_strength_multiplier: scale-factor for the two-body potential term
        :param top_k_noise_std: noise injected to the one-body potential prior to chosing top proposals
        
        .. note::
          the source distribution (pi_t) must be sorted in descending order
          (abundant words come first)
          
        .. note:: at any codebook position, top proposals are chosen according to the
          one-body potential
        """
        
        assert action_policy in {'sampled', 'greedy', 'epsilon-greedy'}
        assert two_body_dropout_policy in {'earlier', 'random'}
        
        pi_t = to_torch(pi_t, device=self.device, dtype=self.dtype)
        if self.assert_shapes:
            assert pi_t.shape == torch.Size((n_types,))

        # assert that pi_t is properly sorted
        assert torch.all((pi_t[:-1] - pi_t[1:]) >= 0.)
        
        # calculate source prior cdf
        pi_cdf_t = torch.cumsum(pi_t, dim=-1)

        # codebook container
        c_btls = torch.zeros(
            (batch_size, n_types, code_length, self.n_symbols),
            device=self.device, dtype=self.dtype)

        # container for log probs (invididual codebook columns)
        log_prob_b1_list = []

        # get the indexed code space
        n_all_codes = self.get_code_space_size(code_length)
        all_codes_als = to_torch(
            self.get_indexed_code_space(code_length),
            device=self.device,
            dtype=self.dtype)
        
        # aux stuff
        top_proposals_per_column = min(top_proposals_per_column, self.get_code_space_size(code_length))
        random_proposals_per_column = min(random_proposals_per_column, self.get_code_space_size(code_length))
        n_types_torch = torch.tensor(n_types, device=self.device, dtype=self.dtype)
        code_length_torch = torch.tensor(code_length, device=self.device, dtype=self.dtype)
        type_rank_t = torch.arange(n_types, device=self.device, dtype=self.dtype) / n_types
        
        # sweep order (from most abundant to least abundant)
        sweep_order = np.arange(n_types)

        for i_p, i_t in enumerate(sweep_order):

            # calculate one-body potential at the current codebook column over all codes
            one_body_a = self.one_body_potential(
                c_bls=all_codes_als,
                n_types=n_types_torch,
                type_rank=type_rank_t[i_t],
                pi=pi_t[i_t],
                pi_cdf=pi_cdf_t[i_t])
            
            # truncate the code space according to the one-body potential
            if top_k_noise_std > 0.:
                noisy_one_body_a = (
                    one_body_a +
                    top_k_noise_std * torch.std(one_body_a) * torch.randn_like(one_body_a))
            else:
                noisy_one_body_a = one_body_a
            trunc_code_indices_k = torch.unique(
                torch.cat(
                    (torch.topk(-noisy_one_body_a, top_proposals_per_column)[1],
                     torch.randint(0, n_all_codes, [random_proposals_per_column], device=self.device)),
                    dim=-1))
            trunc_codes_kls = all_codes_als[trunc_code_indices_k, ...]
            n_trunc_codes = trunc_code_indices_k.shape[0]
            
            # one body potential over truncated codes
            if not disable_one_body:
                one_body_k = one_body_a[trunc_code_indices_k]
            else:
                one_body_k = torch.zeros((n_trunc_codes,), device=self.device)

            # calculate the two-body potential
            if i_p > 0 and not disable_two_body:
                
                # choose columns to interact with
                if two_body_max_n_interactions < (n_types - 1):
                    if two_body_dropout_policy == 'earlier':
                        interaction_indices = sweep_order[max(0, i_p - two_body_max_n_interactions) : i_p]
                    elif two_body_dropout_policy == 'random':
                        interaction_indices = sweep_order[:i_p]\
                            [np.random.permutation(i_p)][:min(i_p, two_body_max_n_interactions)]
                    else:
                        raise ValueError("Bad value for two_body_dropout_policy; allowed values: earlier, random")
                else:
                    interaction_indices = sweep_order[:i_p]
                
                n_interaction = len(interaction_indices)
                
                c_1_bkils = trunc_codes_kls[None, :, None, :, :].expand(
                    (batch_size, n_trunc_codes, n_interaction, code_length, self.n_symbols))
                c_2_bkils = c_btls[:, None, interaction_indices, :, :].expand(
                    (batch_size, n_trunc_codes, n_interaction, code_length, self.n_symbols))
                
                pi_1_bki = pi_t[interaction_indices].expand((batch_size, n_trunc_codes, n_interaction))
                pi_2_bki = pi_t[i_t].expand((batch_size, n_trunc_codes, n_interaction))
                
                dropout_compensation_factor = i_p / n_interaction
                two_body_bk = dropout_compensation_factor * self.two_body_potential.forward(
                    c_1_bls=c_1_bkils,
                    nu_1_bj=None,
                    pi_1_b=pi_1_bki,
                    c_2_bls=c_2_bkils,
                    nu_2_bj=None,
                    pi_2_b=pi_2_bki).sum(-1)

            else:
                two_body_bk = torch.zeros((batch_size, n_trunc_codes), device=self.device)

            # unnormalized conditional probability at column i_t
            unnorm_log_prob_bk = - (
                one_body_strength_multiplier * one_body_k.expand((batch_size, n_trunc_codes)) +
                two_body_strength_multiplier * two_body_bk)
            
            # if hardcore potential is enabled, set the log_prob to -inf on previously used codes
            if i_p > 0 and enable_hardcore_two_body_potential:
                # a binary mask indicating which elements of the truncated code space (k)
                # collides with one or more of the previously sampled codes separately
                # in each codebook instance (b)
                mask_bk = (
                    (trunc_codes_kls[None, :, None, :, :] - c_btls[:, None, :i_p, :, :])
                        .abs().sum((-1, -2)) == 0).sum(-1) > 0
                unnorm_log_prob_bk[mask_bk] = -float("inf")
            
            # normalize the conditional probability at column i_t
            log_prob_bk = unnorm_log_prob_bk - torch.logsumexp(
                unnorm_log_prob_bk, dim=-1, keepdim=True)

            # sample code index
            if action_policy == 'sampled':
                c_b = torch.distributions.Categorical(logits=log_prob_bk).sample()
            elif action_policy == 'greedy':
                c_b = log_prob_bk.argmax(-1)
            elif action_policy == 'epsilon-greedy':
                c_sampled_b = torch.distributions.Categorical(logits=log_prob_bk).sample()
                c_greedy_b = log_prob_bk.argmax(-1)
                c_b = torch.where(
                    torch.rand(batch_size, device=self.device) > action_policy_epsilon,
                    c_greedy_b,
                    c_sampled_b)
            else:
                raise ValueError(
                    f'Bad value for action_policy ({action_policy}); '
                    f'allows values: sampled, greedy, epsilon-greedy')

            # lookup codes and store in the container
            c_btls[:, i_t, :, :] = trunc_codes_kls[c_b, :, :]

            # select log probs
            log_prob_b1_list.append(torch.gather(log_prob_bk, -1, c_b.unsqueeze(-1)))
        
        # sum the log probs of columns to get the codebook log prob
        log_prob_b = torch.cat(log_prob_b1_list, dim=-1).mean(-1)
        
        return {
            'codebook_btls': c_btls,
            'log_prob_b': log_prob_b
        }


def code_space_filter_func_rel_symbol_weight(
        c_als: np.ndarray,
        code_length: int,
        min_rel_symbol_weight_s: np.ndarray,
        max_rel_symbol_weight_s: np.ndarray) -> np.ndarray:
    n_symbols, code_length = c_als.shape[-1], c_als.shape[-2]
    n_as = c_als.sum(-2)
    passing_a = (
        (n_as >= min_rel_symbol_weight_s[None, :] * code_length) &
        (n_as <= max_rel_symbol_weight_s[None, :] * code_length)).sum(-1) == n_symbols
    return passing_a


def code_space_filter_func_abs_hamming_weight(
        c_als: np.ndarray,
        code_length: int,
        min_symbol_weight_s: np.ndarray,
        max_symbol_weight_s: np.ndarray) -> np.ndarray:
    n_symbols, code_length = c_als.shape[-1], c_als.shape[-2]
    n_as = c_als.sum(-2)
    passing_a = (
        (n_as >= min_symbol_weight_s[None, :]) &
        (n_as <= max_symbol_weight_s[None, :])).sum(-1) == n_symbols
    return passing_a