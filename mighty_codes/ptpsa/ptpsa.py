import numpy as np
import math
import torch
import pyro
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, NamedTuple
from enum import Enum
from abc import ABCMeta, abstractmethod
from boltons.cacheutils import cachedproperty, cachedmethod
from collections import deque

from mighty_codes.instr_utils import Timer
from mighty_codes.math_utils import logaddexp
from mighty_codes.torch_utils import to_np


class BatchedStateManipulator(metaclass=ABCMeta):
    
    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype):
        
        self.device = device
        self.dtype = dtype
        
        self.__pre__()
        
    @property
    @abstractmethod
    def m(self) -> int:
        """Cardinality of atomic moves"""
        return NotImplementedError
    
    @property
    @abstractmethod
    def move_types_dict(self) -> Dict[str, int]:
        return NotImplementedError
    
    @property
    @abstractmethod
    def move_type_m(self) -> torch.Tensor:
        return NotImplementedError
    
    @property
    @abstractmethod
    def move_size_m(self) -> torch.Tensor:
        return NotImplementedError
    
    @property
    @abstractmethod
    def move_log_tav_m(self) -> torch.Tensor:
        return NotImplementedError
    
    @abstractmethod
    def perturb(
            self,
            state_bx: torch.Tensor,
            move_type_bk: torch.Tensor,
            move_size_bk: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    
    @abstractmethod
    def generate_random_state_batch(
            self,
            batch_size: int) -> torch.Tensor:
        return NotImplementedError
    
    def __pre__(self):
        for move_name, move_index in self.move_types_dict.items():
            setattr(self, move_name, move_index)
            
    def __post__(self):
        assert 'MOVE_SKIP' in self.move_types_dict
        assert self.move_types_dict['MOVE_SKIP'] == 0
        
        assert self.m > 0
        
        assert self.move_type_m.shape == (self.m,)
        assert self.move_size_m.shape == (self.m,)
        assert self.move_log_tav_m.shape == (self.m,)
        
        move_types_expected_int_set = set(self.move_types_dict.values())
        move_types_provided_int_set = set(self.move_type_m.tolist())
        assert not (0 in move_types_provided_int_set)
        assert all(t in move_types_expected_int_set for t in move_types_provided_int_set)
        

class BatchedStateEnergyCalculator(metaclass=ABCMeta):
    
    @abstractmethod
    def calculate_state_energy(self, state_bx: torch.Tensor) -> torch.Tensor:
        return NotImplementedError
    

class MCMCAcceptanceRatePredictor(torch.nn.Module):
    """Predicts MCMC acceptance rate given temperature, initial energy, and path."""
    
    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype):
        super(MCMCAcceptanceRatePredictor, self).__init__()
        
        self.device = device
        self.dtype = dtype
                
    def forward(
            self,
            pert_nbkm: torch.Tensor,
            log_temperature_b: torch.Tensor,
            initial_energy_b: torch.Tensor) -> torch.Tensor:
        """Returns predicted acceptance probability in logit space."""
        
        raise NotImplementedError


class StateEnergyBundle(NamedTuple):
    state_bx: torch.Tensor
    energy_b: torch.Tensor

class SimulatedAnnealingStateSummary(NamedTuple):

    i_iter: int
    beta: float

    local_energy_min: float
    local_energy_max: float
    local_energy_mean: float
    local_energy_std: float
        
    global_energy_min: float
        
    mcmc_acc_rate_pred: float
    mcmc_acc_rate_emp: float
    mcmc_acc_rate_lambda: float

    log_prob_mcmc_prop_n_moves_k: np.ndarray
    log_prob_mcmc_prop_move_type_m: np.ndarray


class SimulatedAnnealingIterationSummary(NamedTuple):
    
    # iteration index at the end
    i_iter: int
    
    # what performed
    performed_mcmc: bool = False
    performed_cooling: bool = False
    performed_resampling_local: bool = False
    performed_resampling_global: bool = False
    performed_reheating: bool = False
        

class SimulatedAnnealingExitCode(Enum):
    CONTINUE = 0
    REACHED_LOWEST_TEMPERATURE = 1
    REACHED_MAX_ITERS = 2
    REACHED_MAX_CYCLES = 3
    REACHED_ABS_TOL_CONVERGENCE = 4

    
class PyTorchBatchedSimulatedAnnealing:    
    def __init__(
            self,
            batched_state_energy_calculator: BatchedStateEnergyCalculator,
            batched_state_manipulator: BatchedStateManipulator,
            mcmc_acceptance_rate_predictor: MCMCAcceptanceRatePredictor,
            device: torch.device,
            dtype: torch.dtype,
            **kwargs):
        """
        :param state_energy_calculator: a callable function that takes a tensor with shape
          (batch_index, n_types, code_length, n_symbols) and return a tensor with shape
          (batch_index, n_types)
        """
        
        self.device = device
        self.dtype = dtype
        
        # modules
        self.batched_state_energy_calculator = batched_state_energy_calculator
        self.batched_state_manipulator = batched_state_manipulator
        self.mcmc_acceptance_rate_predictor = mcmc_acceptance_rate_predictor
        
        self.quality_factor = kwargs['quality_factor']
        assert self.quality_factor >= 1
        
        self.batch_size = kwargs['n_subsystems']
        assert self.batch_size >= 1
        
        self.n_resampling_groups = kwargs['n_resampling_groups']
        assert self.batch_size % self.n_resampling_groups == 0
    
        self.base_cooling_iters = kwargs['base_cooling_iters']
        assert self.base_cooling_iters > 0.
        
        self.population_buffer_size = kwargs['population_buffer_size']
        assert self.population_buffer_size >= 1
    
        self.resampling_local_interval = kwargs['resampling_local_interval']
        assert self.resampling_local_interval >= 1

        self.resampling_global_interval = kwargs['resampling_global_interval']
        assert self.resampling_global_interval >= 1
        
        self.resampling_start_cycle_position = kwargs['resampling_start_cycle_position']
        assert 0. <= self.resampling_start_cycle_position <= 1.
        
        self.quench_ratio = kwargs['quench_ratio']
        assert 0. < self.quench_ratio < 1.
        
        self.reheating_ratio = kwargs['reheating_ratio']
        assert 0. < self.reheating_ratio < 1.
        
        self.reheating_resampling_beta_ratio = kwargs['reheating_resampling_beta_ratio']
        assert self.reheating_resampling_beta_ratio > 0.
        
        self.mcmc_eff_acc_rate_lo = kwargs['mcmc_eff_acc_rate_lo']
        assert 0. < self.mcmc_eff_acc_rate_lo < 1.

        self.mcmc_eff_acc_rate_hi = kwargs['mcmc_eff_acc_rate_hi']
        assert 0. < self.mcmc_eff_acc_rate_hi < 1.
        
        self.rand_state_energy_loc = kwargs['rand_state_energy_loc']
        self.rand_state_energy_scale = kwargs['rand_state_energy_scale']
        assert self.rand_state_energy_scale > 0.
        
        self.dimensionless_beta_0 = kwargs['dimensionless_beta_0']
        self.dimensionless_beta_f = kwargs['dimensionless_beta_f']
        self.dimensionless_beta_max = kwargs['dimensionless_beta_max']
        assert self.dimensionless_beta_0 > 0.
        assert self.dimensionless_beta_f >- self.dimensionless_beta_f
        assert self.dimensionless_beta_max >= self.dimensionless_beta_f
        self.beta_0 = self.dimensionless_beta_0 / self.rand_state_energy_scale
        self.beta_f = self.dimensionless_beta_f / self.rand_state_energy_scale
        self.beta_max = self.dimensionless_beta_max / self.rand_state_energy_scale
        
        self.perturb_max_moves = kwargs['perturb_max_moves']
        assert self.perturb_max_moves >= 1
        
        # moving average forgetting factors
        self.prob_mcmc_acc_emp_ma_beta = kwargs['prob_mcmc_acc_emp_ma_beta']
        self.prob_mcmc_acc_pred_ma_beta = kwargs['prob_mcmc_acc_pred_ma_beta']
        assert 0. <= self.prob_mcmc_acc_emp_ma_beta < 1.
        assert 0. <= self.prob_mcmc_acc_pred_ma_beta < 1.
        
        # learning acceptance rate
        self.mcmc_acc_pred_optim_lr = kwargs['mcmc_acc_pred_optim_lr']
        self.mcmc_acc_pred_interval = kwargs['mcmc_acc_pred_interval']
        assert self.mcmc_acc_pred_optim_lr > 0.
        assert self.mcmc_acc_pred_interval >= 1
        
        # mcmc proposal learning
        self.mcmc_prop_n_moves_uniform_admix = kwargs['mcmc_prop_n_moves_uniform_admix']
        self.mcmc_prop_move_types_uniform_admix = kwargs['mcmc_prop_move_types_uniform_admix']
        self.mcmc_prop_optim_lr = kwargs['mcmc_prop_optim_lr']
        self.n_mcmc_prop_optim_steps_after_reheating = kwargs['n_mcmc_prop_optim_steps_after_reheating']
        self.n_mcmc_prop_optim_steps_during_cooling = kwargs['n_mcmc_prop_optim_steps_during_cooling']
        assert 0. <= self.mcmc_prop_n_moves_uniform_admix <= 1.
        assert 0. <= self.mcmc_prop_move_types_uniform_admix <= 1.
        assert self.mcmc_prop_optim_lr > 0.
        assert self.n_mcmc_prop_optim_steps_after_reheating >= 0
        assert self.n_mcmc_prop_optim_steps_during_cooling >= 1
        
        # miscellaneous
        self.gumble_softmax_temperature = kwargs['gumble_softmax_temperature']
        assert self.gumble_softmax_temperature > 0.
        
        self.n_path_samples_per_subsystem = kwargs['n_path_samples_per_subsystem']
        assert self.n_path_samples_per_subsystem >= 1
        
        self.lambda_mcmc_acc_rate_lo = kwargs['lambda_mcmc_acc_rate_lo']
        self.lambda_mcmc_acc_rate_hi = kwargs['lambda_mcmc_acc_rate_hi']
        self.lambda_mcmc_acc_decay_rate = kwargs['lambda_mcmc_acc_decay_rate']
        self.lambda_mcmc_acc_increase_rate = kwargs['lambda_mcmc_acc_increase_rate']
        assert self.lambda_mcmc_acc_rate_lo > 0.
        assert self.lambda_mcmc_acc_rate_hi > self.lambda_mcmc_acc_rate_lo
        assert self.lambda_mcmc_acc_decay_rate >= 0.
        assert self.lambda_mcmc_acc_increase_rate >= 0.
        
        self.log_prob_mcmc_prop_move_type_m_unconstrained_range = \
            kwargs['log_prob_mcmc_prop_move_type_m_unconstrained_range']
        self.log_prob_mcmc_prop_n_moves_k_unconstrained_range = \
            kwargs['log_prob_mcmc_prop_n_moves_k_unconstrained_range']
        assert self.log_prob_mcmc_prop_move_type_m_unconstrained_range > 0.
        assert self.log_prob_mcmc_prop_n_moves_k_unconstrained_range > 0.
        
        self.top_k_keep = kwargs['top_k_keep']
        self.max_iters = kwargs['max_iters']
        self.max_cycles = kwargs['max_cycles']
        self.convergence_abs_tol = float(kwargs['convergence_abs_tol'])
        self.convergence_countdown = int(kwargs['convergence_countdown'])
        assert self.top_k_keep >= 1
        assert self.max_iters >= 1
        assert self.max_cycles >= 1
        assert self.convergence_abs_tol > 0.
        assert self.convergence_countdown >= 1
        
        self.record_iteration_summary = kwargs['record_iteration_summary']
        self.record_state_summary = kwargs['record_state_summary']

        # other choices (hardcoded)
        self.mcmc_prop_optim_class = torch.optim.Adamax
        self.mcmc_acc_pred_optim_class = torch.optim.Adamax

        # initialize SA variables
        self.prob_mcmc_acc_emp = 0.
        self.prob_mcmc_acc_pred = 0.
        self.beta = self.beta_0
        self.curr_cooling_cycle_beta_0 = self.beta_0
        self.curr_cooling_cycle_beta_f = min(self.beta_max, self.curr_cooling_cycle_beta_0 / self.quench_ratio)
        self.cooling_rate = self._get_cooling_rate()
    
        # initialize to random codebooks
        self.state_bx = self.batched_state_manipulator.generate_random_state_batch(
            batch_size=self.batch_size)
        
        # initialize per-type energies
        self.energy_b = self.batched_state_energy_calculator.calculate_state_energy(
            state_bx=self.state_bx)
        
        # MCMC number of moves
        self.log_prob_mcmc_prop_n_moves_k_unconstrained = torch.nn.Parameter(
            torch.zeros(
                (self.perturb_max_moves,),
                device=device, dtype=dtype))
        
        # MCMC move type
        self.log_prob_mcmc_prop_move_type_m_unconstrained = torch.nn.Parameter(
            torch.zeros(
                (self.batched_state_manipulator.m,),
                device=device, dtype=dtype))
        
        # dynamical penalty for violating MCMC accept rate
        self.log_lambda_mcmc_acc_rate = torch.nn.Parameter(
            torch.tensor(
                np.log(self.lambda_mcmc_acc_rate_lo),
                device=self.device, dtype=self.dtype))
        
        # optimizer for learning MCMC proposal probability
        self.mcmc_prop_optim = self.mcmc_prop_optim_class(
            params=[
                self.log_prob_mcmc_prop_n_moves_k_unconstrained,
                self.log_prob_mcmc_prop_move_type_m_unconstrained,
                self.log_lambda_mcmc_acc_rate],
            lr=self.mcmc_prop_optim_lr)

        # optimizer for learning MCMC proposal acceptance rate
        self.mcmc_acc_pred_optim = self.mcmc_acc_pred_optim_class(
            params=list(self.mcmc_acceptance_rate_predictor.parameters()),
            lr=self.mcmc_acc_pred_optim_lr)
        
        # top states
        self.top_state_kx = self.batched_state_manipulator.generate_random_state_batch(
            batch_size=self.top_k_keep)
        self.top_energy_k = float("inf") * torch.ones(
            (self.top_k_keep,), device=device, dtype=dtype)
        self.top_energy = float("inf")
        self.prev_cycle_top_energy = float("inf")
        
        # initialize resampling population buffer
        self.resampling_buffer = deque([], maxlen=self.population_buffer_size)

        # book-keeping
        
        self.i_iter = 0
        self.i_cycle = 0
        self.i_below_convergence_abs_tol = 0
        
        self.iteration_summary_list: List[SimulatedAnnealingIterationSummary] = []
        self.state_summary_list: List[SimulatedAnnealingStateSummary] = []

        self.timer_dict = {
            'mcmc_proposal_generation': Timer(),
            'mcmc_proposal_evaluation': Timer(),
            'mcmc_acc_rate_predictor_update': Timer(),
            'mcmc_prop_update': Timer(),
            'mcmc_bookkeeping': Timer(),
            'cooling': Timer(),
            'resampling': Timer(),
            'reheating': Timer(),
            'full_iteration': Timer(),
        }

    def _get_cooling_rate(self) -> float:
        """Calculate cooling rate for the current cooling cycle"""
        effective_quench_ratio = self.curr_cooling_cycle_beta_0 / self.curr_cooling_cycle_beta_f
        return (np.exp(-np.log(effective_quench_ratio) / (self.quality_factor * self.base_cooling_iters))
                - 1.0) / self.mcmc_eff_acc_rate_lo

    @cachedproperty
    def local_resampling_grouping_log_prob_bb(self) -> torch.tensor:
        mask_bb = torch.zeros((self.batch_size, self.batch_size), device=self.device, dtype=self.dtype)
        mask_bb.fill_(-float("inf"))
        resampling_group_size = self.batch_size // self.n_resampling_groups
        for i_group in range(self.n_resampling_groups):
            i_begin = i_group * resampling_group_size
            i_end = (i_group + 1) * resampling_group_size
            mask_bb[i_begin:i_end, i_begin:i_end] = 0.
        return mask_bb
    
    @property
    def prob_mcmc_acc_emp_unbiased(self) -> float:
        if self.i_iter > 0:
            return self.prob_mcmc_acc_emp / (1. - self.prob_mcmc_acc_emp_ma_beta ** self.i_iter)
        else:
            return 0.
    
    @property
    def prob_mcmc_acc_pred_unbiased(self) -> float:
        if self.i_iter > 0:
            return self.prob_mcmc_acc_pred / (1. - self.prob_mcmc_acc_pred_ma_beta ** self.i_iter)
        else:
            return 0.

    @property
    def log_prob_mcmc_prop_n_moves_k(self) -> torch.Tensor:
        log_prob_mcmc_prop_n_moves_k = self.log_prob_mcmc_prop_n_moves_k_unconstrained \
            - torch.logsumexp(self.log_prob_mcmc_prop_n_moves_k_unconstrained, dim=-1, keepdim=True)
        log_prob_mcmc_prop_n_moves_uniform_k = - np.log(self.perturb_max_moves) * torch.ones_like(
            log_prob_mcmc_prop_n_moves_k)
        return logaddexp(
            log_prob_mcmc_prop_n_moves_uniform_k + np.log(self.mcmc_prop_n_moves_uniform_admix),
            log_prob_mcmc_prop_n_moves_k + np.log(1. - self.mcmc_prop_n_moves_uniform_admix))
    
    @property
    def log_prob_mcmc_prop_move_type_m(self) -> torch.Tensor:
        log_prob_mcmc_prop_move_type_m = self.log_prob_mcmc_prop_move_type_m_unconstrained \
            - torch.logsumexp(self.log_prob_mcmc_prop_move_type_m_unconstrained, dim=-1, keepdim=True)
        log_prob_mcmc_prop_indexed_move_uniform_m = - np.log(self.batched_state_manipulator.m) * torch.ones_like(
            log_prob_mcmc_prop_move_type_m)
        return logaddexp(
            log_prob_mcmc_prop_indexed_move_uniform_m + np.log(self.mcmc_prop_move_types_uniform_admix),
            log_prob_mcmc_prop_move_type_m + np.log(1. - self.mcmc_prop_move_types_uniform_admix))

    def propose_mcmc_move(self):
        log_prob_mcmc_prop_n_moves_bk = self.log_prob_mcmc_prop_n_moves_k.expand(
            [self.batch_size, self.perturb_max_moves])
        log_prob_mcmc_prop_indexed_move_bm = self.log_prob_mcmc_prop_move_type_m.expand(
            [self.batch_size, self.batched_state_manipulator.m])
        n_moves_b = 1 + torch.distributions.Categorical(
            logits=log_prob_mcmc_prop_n_moves_bk).sample()
        move_index_bk = torch.distributions.Categorical(
            logits=log_prob_mcmc_prop_indexed_move_bm[:, None, :].expand(
                (self.batch_size, self.perturb_max_moves, self.batched_state_manipulator.m))).sample()
        pert_type_bk = self.batched_state_manipulator.move_type_m[move_index_bk]
        pert_size_bk = self.batched_state_manipulator.move_size_m[move_index_bk]
        skip_mask_bk = torch.arange(self.perturb_max_moves, device=self.device)[None, :] >= n_moves_b[:, None]
        pert_type_bk[skip_mask_bk] = self.batched_state_manipulator.MOVE_SKIP
        return {
            'pert_type_bk': pert_type_bk,
            'pert_size_bk': pert_size_bk,
            'move_index_bk': move_index_bk,
            'n_moves_b': n_moves_b,
            'skip_mask_bk': skip_mask_bk}
        
    def perform_mcmc_update(self) -> SimulatedAnnealingIterationSummary:        
        with self.timer_dict['mcmc_proposal_generation']:
            # get a move proposal for each chain
            mcmc_prop_move_dict = self.propose_mcmc_move()

            # generate proposals
            old_state_bx = self.state_bx
            new_state_bx = self.batched_state_manipulator.perturb(
                state_bx=old_state_bx,
                pert_type_bk=mcmc_prop_move_dict['pert_type_bk'],
                pert_size_bk=mcmc_prop_move_dict['pert_size_bk'])
        
        with self.timer_dict['mcmc_proposal_evaluation']:
            # calculate energies
            new_energy_b = self.batched_state_energy_calculator.calculate_state_energy(new_state_bx)
        
            # calculate Boltzmann factor
            old_energy_b = self.energy_b
            log_boltzmann_factor_b = -self.beta * (new_energy_b - old_energy_b)
            accept_mask_b = log_boltzmann_factor_b > torch.rand_like(new_energy_b).log()

        with self.timer_dict['mcmc_acc_rate_predictor_update']:
            # update MCMC acceptance rate predictor
            self._perform_mcmc_acceptance_rate_predictor_update(
                accept_mask_b=accept_mask_b,
                move_index_bk=mcmc_prop_move_dict['move_index_bk'],
                skip_mask_bk=mcmc_prop_move_dict['skip_mask_bk'])

        with self.timer_dict['mcmc_bookkeeping']:
            # update states and energies
            self.state_bx = torch.where(
                accept_mask_b[:, None],
                new_state_bx,
                old_state_bx).clone()
            self.energy_b = torch.where(
                accept_mask_b,
                new_energy_b,
                old_energy_b).clone()
            
            # update empirical accept rates of all subsystems
            prob_mcmc_acc_emp_point_est = accept_mask_b.sum().item() / self.batch_size
            self.prob_mcmc_acc_emp = \
                (1. - self.prob_mcmc_acc_emp_ma_beta) * prob_mcmc_acc_emp_point_est + \
                self.prob_mcmc_acc_emp_ma_beta * self.prob_mcmc_acc_emp

            # update top-k solutions
            self._update_top_k()
        
        with self.timer_dict['mcmc_prop_update']:
            # update move suggestion probabilities
            for _ in range(self.n_mcmc_prop_optim_steps_during_cooling):
                self._perform_mcmc_prop_update()
        
        return SimulatedAnnealingIterationSummary(
            i_iter=self.i_iter,
            performed_mcmc=True)
        
    def _update_population_buffer(self):
        self.resampling_buffer.append(
            StateEnergyBundle(
                state_bx=self.state_bx.clone(),
                energy_b=self.energy_b.clone()))
        
    def _update_top_k(self):
        all_state_zx = torch.cat(
            (self.top_state_kx, self.state_bx), dim=0)
        all_energy_z = torch.cat(
            (self.top_energy_k, self.energy_b), dim=0)
        
        top_k_indices = torch.argsort(all_energy_z)[:self.top_k_keep]
        self.top_state_kx = all_state_zx[top_k_indices, :].clone()
        self.top_energy_k = all_energy_z[top_k_indices].clone()
        self.top_energy = self.top_energy_k[0].item()

    def _perform_mcmc_acceptance_rate_predictor_update(
            self,
            accept_mask_b: torch.Tensor,
            move_index_bk: torch.Tensor,
            skip_mask_bk: torch.Tensor):
        
        if self.i_iter % self.mcmc_acc_pred_interval != 0:
            return
        
        # zero gradient info
        self.mcmc_acc_pred_optim.zero_grad()
        
        # generate input
        pert_bkm = (
            torch.eye(self.batched_state_manipulator.m, device=self.device, dtype=self.dtype)[move_index_bk] *
            (~skip_mask_bk).type(self.dtype).unsqueeze(-1))
        
        # predict
        initial_energy_b = self.energy_b
        log_temperature_b = - np.log(self.beta) * torch.ones_like(initial_energy_b)
        logit_accept_b = self.mcmc_acceptance_rate_predictor.forward(
            pert_nbkm=pert_bkm[None, ...],
            log_temperature_b=log_temperature_b,
            initial_energy_b=initial_energy_b).squeeze(0)

        # loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit_accept_b,
            accept_mask_b.to(self.dtype),
            reduction='sum')
        
        # update
        loss.backward()
        self.mcmc_acc_pred_optim.step()
        
        # bookkeeping
        with torch.no_grad():
            pred_accept_mask_b = torch.sigmoid(logit_accept_b)
            prob_mcmc_acc_emp_point_est = pred_accept_mask_b.sum().item() / self.batch_size
            self.prob_mcmc_acc_pred = \
                (1. - self.prob_mcmc_acc_pred_ma_beta) * prob_mcmc_acc_emp_point_est + \
                self.prob_mcmc_acc_pred_ma_beta * self.prob_mcmc_acc_pred
    
    @cachedproperty
    def _inc_mask_kk(self) -> torch.Tensor:
        return (torch.arange(0, self.perturb_max_moves)[:, None] >=
                torch.arange(0, self.perturb_max_moves)[None, :]).to(self.device).type(self.dtype)
        
    def _perform_mcmc_prop_update(self):
        
        # zero gradient info
        self.mcmc_prop_optim.zero_grad()

        # sample paths
        log_prob_mcmc_prop_n_moves_bk = self.log_prob_mcmc_prop_n_moves_k.expand(
            [self.batch_size, self.perturb_max_moves])
        log_prob_mcmc_prop_indexed_move_bm = self.log_prob_mcmc_prop_move_type_m.expand(
            [self.batch_size, self.batched_state_manipulator.m])
        n_moves_nbk = pyro.distributions.RelaxedOneHotCategoricalStraightThrough(
            temperature=self.gumble_softmax_temperature,
            logits=log_prob_mcmc_prop_n_moves_bk[None, :, :].expand(
                [self.n_path_samples_per_subsystem, self.batch_size, self.perturb_max_moves])).rsample()
        move_index_nbkm = pyro.distributions.RelaxedOneHotCategoricalStraightThrough(
            temperature=self.gumble_softmax_temperature,
            logits=log_prob_mcmc_prop_indexed_move_bm[None, :, None, :].expand(
                (self.n_path_samples_per_subsystem, self.batch_size,
                 self.perturb_max_moves, self.batched_state_manipulator.m))).rsample()
        
        # predict accept rate
        inc_mask_nbk = torch.einsum("nbk,kq->nbq", n_moves_nbk, self._inc_mask_kk)
        pert_nbkm = move_index_nbkm * inc_mask_nbk[..., None]
        initial_energy_b = self.energy_b
        log_temperature_b = - np.log(self.beta) * torch.ones_like(initial_energy_b)
        logit_accept_nb = self.mcmc_acceptance_rate_predictor.forward(
            pert_nbkm=pert_nbkm,
            log_temperature_b=log_temperature_b,
            initial_energy_b=initial_energy_b)
        prob_accept_nb = torch.sigmoid(logit_accept_nb)
        
        # loss
        prob_accept_mean = prob_accept_nb.mean()
        prob_accept_deviation = (
            torch.clamp(prob_accept_mean - self.mcmc_eff_acc_rate_hi, min=0.) +
            torch.clamp(self.mcmc_eff_acc_rate_lo - prob_accept_mean, min=0.))
        prob_accept_in_range = (prob_accept_mean < self.mcmc_eff_acc_rate_hi) & (prob_accept_mean > self.mcmc_eff_acc_rate_lo)
        prob_accept_in_range = prob_accept_in_range.type(self.dtype).clone().detach()
        lambda_rate_of_change = (
            (1. - prob_accept_in_range) * self.lambda_mcmc_acc_increase_rate
            - prob_accept_in_range * self.lambda_mcmc_acc_decay_rate) 
        
        #         # mean target accessible volume for the sampled paths
        #         log_tav_m = self.m_to_log_tav
        #         log_tav_nbk = torch.sum(move_index_nbkm * log_tav_m, -1)
        #         log_tav_nb = torch.sum(n_moves_nbk * torch.cumsum(log_tav_nbk, -1), -1)
        #         log_prob_accept_nb = torch.nn.functional.logsigmoid(logit_accept_nb)
        #         log_tav_b = torch.logsumexp(log_tav_nb + log_prob_accept_nb, dim=0) - np.log(self.n_path_samples_per_subsystem)
        #         log_tav = torch.logsumexp(log_tav_b, dim=0) - np.log(self.batch_size)

        #         mean_n_moves = (
        #             log_prob_mcmc_prop_n_moves_bk.exp() *
        #             torch.arange(1, self.perturb_max_moves + 1, device=self.device, dtype=self.dtype)).sum(-1).mean()

        move_type_entropy = - (self.log_prob_mcmc_prop_move_type_m * self.log_prob_mcmc_prop_move_type_m.exp()).sum() \
            / np.log(self.batched_state_manipulator.m)
        
        # maximize tav subj. prob acc being in a given range
        lambda_mcmc_acc_rate = self.log_lambda_mcmc_acc_rate.exp()
        loss = (
            - move_type_entropy
            + lambda_mcmc_acc_rate.clone().detach() * prob_accept_deviation
            - lambda_mcmc_acc_rate * lambda_rate_of_change)
        
        # gradient update
        loss.backward()
        self.mcmc_prop_optim.step()
        
        # clamp ranges
        with torch.no_grad():
            # lambda
            self.log_lambda_mcmc_acc_rate.data = torch.clamp(
                self.log_lambda_mcmc_acc_rate.data,
                min=np.log(self.lambda_mcmc_acc_rate_lo),
                max=np.log(self.lambda_mcmc_acc_rate_hi))
            # move_type
            self.log_prob_mcmc_prop_move_type_m_unconstrained.data = torch.clamp(
                self.log_prob_mcmc_prop_move_type_m_unconstrained.data -
                self.log_prob_mcmc_prop_move_type_m_unconstrained.data.mean(),
                min=-self.log_prob_mcmc_prop_move_type_m_unconstrained_range,
                max=self.log_prob_mcmc_prop_move_type_m_unconstrained_range)
            # n_moves
            self.log_prob_mcmc_prop_n_moves_k_unconstrained.data = torch.clamp(
                self.log_prob_mcmc_prop_n_moves_k_unconstrained.data -
                self.log_prob_mcmc_prop_n_moves_k_unconstrained.data.mean(),
                min=-self.log_prob_mcmc_prop_n_moves_k_unconstrained_range,
                max=self.log_prob_mcmc_prop_n_moves_k_unconstrained_range)
    
    def perform_cooling_update(self) -> Optional[SimulatedAnnealingStateSummary]:
        if self.beta >= self.curr_cooling_cycle_beta_f:
            return None
        
        self.beta *= (1. + self.cooling_rate * self.prob_mcmc_acc_emp_unbiased)

        return SimulatedAnnealingIterationSummary(
            i_iter=self.i_iter,
            performed_cooling=True)
  
    def perform_reheating_update(self) -> Optional[SimulatedAnnealingStateSummary]:
        with self.timer_dict['reheating']:
            if self.beta < self.curr_cooling_cycle_beta_f:
                return None

            # add current population to buffer
            self._update_population_buffer()

            # heat and update cooling rate
            next_cooling_cycle_beta_0 = self.curr_cooling_cycle_beta_0 / self.reheating_ratio
            next_cooling_cycle_beta_f = min(self.beta_max, next_cooling_cycle_beta_0 / self.quench_ratio)

            # select new population
            buffered_energy_n = torch.cat(
                [t.energy_b for t in self.resampling_buffer], 0)
            buffered_state_nx =  torch.cat(
                [t.state_bx for t in self.resampling_buffer], 0)
            n_states = buffered_energy_n.shape[0]

            # bolzmann resampling w/ replacement
            resampling_beta = next_cooling_cycle_beta_0 / self.reheating_resampling_beta_ratio
            log_boltzmann_n = - resampling_beta * buffered_energy_n
            indices = torch.distributions.Categorical(logits=log_boltzmann_n.expand([self.batch_size, n_states])).sample()

            # update state and energy
            self.state_bx = buffered_state_nx[indices, :].clone()
            self.energy_b = buffered_energy_n[indices].clone()

            # setup the next cooling cycle
            self.curr_cooling_cycle_beta_0 = next_cooling_cycle_beta_0
            self.curr_cooling_cycle_beta_f = next_cooling_cycle_beta_f
            self.cooling_rate = self._get_cooling_rate()
            self.beta = next_cooling_cycle_beta_0

            # reset optimizer and proposal state
            self.log_prob_mcmc_prop_move_type_m_unconstrained.data[:] = 0.
            self.log_prob_mcmc_prop_n_moves_k_unconstrained.data[:] = 0.
            self.log_lambda_mcmc_acc_rate.data = torch.tensor(
                np.log(self.lambda_mcmc_acc_rate_lo),
                device=self.device, dtype=self.dtype)
            self.mcmc_prop_optim = self.mcmc_prop_optim_class(
                params=[
                    self.log_prob_mcmc_prop_n_moves_k_unconstrained,
                    self.log_prob_mcmc_prop_move_type_m_unconstrained,
                    self.log_lambda_mcmc_acc_rate],
                lr=self.mcmc_prop_optim_lr)
            for _ in range(self.n_mcmc_prop_optim_steps_after_reheating):
                self._perform_mcmc_prop_update()

        # increment cycle counter
        self.i_cycle += 1
        
        # check for convergence
        cycle_to_cycle_energy_drop = np.abs(self.top_energy - self.prev_cycle_top_energy)
        if cycle_to_cycle_energy_drop < self.convergence_abs_tol:
            self.i_below_convergence_abs_tol += 1
        else:
            self.i_below_convergence_abs_tol = 0
        self.prev_cycle_top_energy = self.top_energy
        
        return SimulatedAnnealingIterationSummary(
            i_iter=self.i_iter,
            performed_reheating=True)
        
    def perform_resampling(self) -> Optional[SimulatedAnnealingStateSummary]:
        with self.timer_dict['resampling']:
            # only resample after a period of non-resampled evolution
            cooling_cycle_beta_width = self.curr_cooling_cycle_beta_f - self.curr_cooling_cycle_beta_0
            position_in_cycle = (self.beta - self.curr_cooling_cycle_beta_0) / cooling_cycle_beta_width
            if position_in_cycle < self.resampling_start_cycle_position:
                return None

            global_resampling_flag = False
            local_resampling_flag = False
            if self.i_iter % (self.quality_factor * self.resampling_global_interval) == 0:    
                log_boltzmann_b = - self.beta * self.energy_b
                log_boltzmann_grouped_bb = log_boltzmann_b.expand(
                    [self.batch_size, self.batch_size])
                global_resampling_flag = True
            elif self.i_iter % (self.quality_factor * self.resampling_local_interval) == 0:
                log_boltzmann_b = - self.beta * self.energy_b
                log_boltzmann_grouped_bb = log_boltzmann_b.expand(
                    [self.batch_size, self.batch_size]) + self.local_resampling_grouping_log_prob_bb
                local_resampling_flag = True
            else:
                return None

            # update codebook
            resampled_indices = torch.distributions.Categorical(logits=log_boltzmann_grouped_bb).sample()
            self.state_bx = self.state_bx[resampled_indices, :].clone()
            self.energy_b = self.energy_b[resampled_indices].clone()

            return SimulatedAnnealingIterationSummary(
                i_iter=self.i_iter,
                performed_resampling_local=local_resampling_flag,
                performed_resampling_global=global_resampling_flag)
    
    @property
    def curr_state_summary(self) -> SimulatedAnnealingStateSummary:
        energy_b = self.energy_b
        return SimulatedAnnealingStateSummary(
            i_iter=self.i_iter,
            beta=self.beta,
            local_energy_min=energy_b.min().item(),
            local_energy_max=energy_b.max().item(),
            local_energy_mean=energy_b.mean().item(),
            local_energy_std=energy_b.std().item(),
            global_energy_min=self.top_energy,
            mcmc_acc_rate_pred=self.prob_mcmc_acc_pred_unbiased,
            mcmc_acc_rate_emp=self.prob_mcmc_acc_emp_unbiased,
            mcmc_acc_rate_lambda=self.log_lambda_mcmc_acc_rate.exp().item(),
            log_prob_mcmc_prop_n_moves_k=to_np(self.log_prob_mcmc_prop_n_moves_k),
            log_prob_mcmc_prop_move_type_m=to_np(self.log_prob_mcmc_prop_move_type_m))
    
    def step(self) -> SimulatedAnnealingExitCode:
        with self.timer_dict['full_iteration']:
            # metropolis-hastings update
            mcmc_out = self.perform_mcmc_update()

            # resampling
            resampling_out = self.perform_resampling()

            # cooling
            cooling_out = self.perform_cooling_update()

            # cooling
            reheating_out = self.perform_reheating_update()

            # update iter counter
            self.i_iter += 1

            # record
            if self.record_iteration_summary:
                if mcmc_out:
                    self.iteration_summary_list.append(mcmc_out)
                if resampling_out:
                    self.iteration_summary_list.append(resampling_out)
                if cooling_out:
                    self.iteration_summary_list.append(cooling_out)
                if reheating_out:
                    self.iteration_summary_list.append(reheating_out)

            if self.record_state_summary:
                self.state_summary_list.append(self.curr_state_summary)
            
            # convergence check
            if self.curr_cooling_cycle_beta_0 >= self.beta_f:
                return SimulatedAnnealingExitCode.REACHED_LOWEST_TEMPERATURE
            elif self.i_iter >= self.max_iters:
                return SimulatedAnnealingExitCode.REACHED_MAX_ITERS
            elif self.i_cycle >= self.max_cycles:
                return SimulatedAnnealingExitCode.REACHED_MAX_CYCLES
            elif self.i_below_convergence_abs_tol >= self.convergence_countdown:
                return SimulatedAnnealingExitCode.REACHED_ABS_TOL_CONVERGENCE
            else:
                return SimulatedAnnealingExitCode.CONTINUE

    def state_dict(self) -> Dict[str, Any]:
        state_dict = dict()

        # children modules
        state_dict['mcmc_acceptance_rate_predictor.state_dict'] = \
            self.mcmc_acceptance_rate_predictor.state_dict()
        state_dict['mcmc_prop_optim.state_dict'] = \
            self.mcmc_prop_optim.state_dict()
        state_dict['mcmc_acc_pred_optim.state_dict'] = \
            self.mcmc_acc_pred_optim.state_dict()

        # parameters
        state_dict['log_prob_mcmc_prop_n_moves_k_unconstrained.data'] = \
            self.log_prob_mcmc_prop_n_moves_k_unconstrained.data
        state_dict['log_prob_mcmc_prop_move_type_m_unconstrained.data'] = \
            self.log_prob_mcmc_prop_move_type_m_unconstrained.data
        state_dict['log_lambda_mcmc_acc_rate.data'] = \
            self.log_lambda_mcmc_acc_rate.data

        # mutable attributes
        state_dict['prob_mcmc_acc_emp'] = self.prob_mcmc_acc_emp
        state_dict['prob_mcmc_acc_pred'] = self.prob_mcmc_acc_pred
        state_dict['beta'] = self.beta
        state_dict['curr_cooling_cycle_beta_0'] = self.curr_cooling_cycle_beta_0
        state_dict['curr_cooling_cycle_beta_f'] = self.curr_cooling_cycle_beta_f
        state_dict['cooling_rate'] = self.cooling_rate

        state_dict['state_bx'] = self.state_bx
        state_dict['energy_b'] = self.energy_b

        state_dict['top_state_kx'] = self.top_state_kx
        state_dict['top_energy_k'] = self.top_energy_k
        state_dict['top_energy'] = self.top_energy

        state_dict['resampling_buffer'] = self.resampling_buffer

        state_dict['i_iter'] = self.i_iter
        state_dict['i_cycle'] = self.i_cycle

        state_dict['iteration_summary_list'] = self.iteration_summary_list
        state_dict['state_summary_list'] = self.state_summary_list

        state_dict['timer_dict'] = self.timer_dict

        return state_dict


    def load_state_dict(self, state_dict: Dict[str, Any]):
        # children modules
        self.mcmc_acceptance_rate_predictor.load_state_dict(
            state_dict['mcmc_acceptance_rate_predictor.state_dict'])
        self.mcmc_prop_optim.load_state_dict(
            state_dict['mcmc_prop_optim.state_dict'])
        self.mcmc_acc_pred_optim.load_state_dict(
            state_dict['mcmc_acc_pred_optim.state_dict'])

        # parameters
        self.log_prob_mcmc_prop_n_moves_k_unconstrained.data = \
            state_dict['log_prob_mcmc_prop_n_moves_k_unconstrained.data']
        self.log_prob_mcmc_prop_move_type_m_unconstrained.data = \
            state_dict['log_prob_mcmc_prop_move_type_m_unconstrained.data']
        self.log_lambda_mcmc_acc_rate.data = \
            state_dict['log_lambda_mcmc_acc_rate.data']      

        # mutable attributes
        self.prob_mcmc_acc_emp = state_dict['prob_mcmc_acc_emp']
        self.prob_mcmc_acc_pred = state_dict['prob_mcmc_acc_pred']
        self.beta = state_dict['beta']
        self.curr_cooling_cycle_beta_0 = state_dict['curr_cooling_cycle_beta_0']
        self.curr_cooling_cycle_beta_f = state_dict['curr_cooling_cycle_beta_f']
        self.cooling_rate = state_dict['cooling_rate']

        self.state_bx = state_dict['state_bx']
        self.energy_b = state_dict['energy_b']

        self.top_state_kx = state_dict['top_state_kx']
        self.top_energy_k = state_dict['top_energy_k']
        self.top_energy = state_dict['top_energy']

        self.resampling_buffer = state_dict['resampling_buffer']

        self.i_iter = state_dict['i_iter']
        self.i_cycle = state_dict['i_cycle']

        self.iteration_summary_list = state_dict['iteration_summary_list']
        self.state_summary_list = state_dict['state_summary_list']

        self.timer_dict = state_dict['timer_dict']
        
    def print_times(self):
        for key, timer in self.timer_dict.items():
            print(f'{key}: {(1000 * timer.total / ptsa.i_iter):.3f}ms')


def estimate_energy_scale(
        n_rounds: int,
        batch_size: int,
        batched_state_energy_calculator: BatchedStateEnergyCalculator,
        batched_state_manipulator: BatchedStateManipulator,
        device: torch.device,
        dtype: torch.dtype) -> Dict[str, float]:

    energies_b_list = []
    for i_round in range(n_rounds):
        random_state_bx = batched_state_manipulator.generate_random_state_batch(
            batch_size=batch_size)
        energies_b = batched_state_energy_calculator.calculate_state_energy(random_state_bx)
        energies_b_list.append(energies_b)

    all_energies_b = torch.cat(energies_b_list)
    
    return {
        'energy_loc': all_energies_b.mean().item(),
        'energy_scale': all_energies_b.std().item()
    }
