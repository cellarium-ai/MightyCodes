import os
import tarfile
from ruamel_yaml import YAML
from ruamel_yaml.comments import CommentedMap
import logging
import pprint
import time
import math

import numpy as np
import torch
from boltons.cacheutils import cachedproperty
from typing import Dict, Any, Callable, Optional

from mighty_codes import consts
from mighty_codes import metric_utils
from mighty_codes import experiments

from mighty_codes.ptpsa import \
    BatchedStateManipulator, \
    BatchedStateEnergyCalculator, \
    MCMCAcceptanceRatePredictor, \
    PyTorchBatchedSimulatedAnnealing, \
    SimulatedAnnealingExitCode

from mighty_codes.torch_utils import \
    to_np, \
    to_torch, \
    to_one_hot_encoded

from mighty_codes.nn_utils import \
    generate_dense_nnet

from mighty_codes.channel_utils import \
    calculate_bac_standard_metric_dict, \
    calculate_bac_f1_reject_auc_metric_dict

from mighty_codes.experiments import \
    ChannelModelSpecification, \
    ExperimentSpecification

from mighty_codes.channels import \
    BinaryChannelSpecification, \
    BinaryAsymmetricChannelModel

from mighty_codes.ptpsa import \
    PyTorchBatchedSimulatedAnnealing, \
    SimulatedAnnealingExitCode, \
    estimate_energy_scale

from mighty_codes.plot_utils import \
    plot_sa_trajectory, \
    plot_sa_codebook, \
    plot_sa_resampling_buffer

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

log_info = logging.info


class BatchedBinaryCodebookManipulator(BatchedStateManipulator):
    
    def __init__(
            self,
            code_length: int,
            n_types: int,
            min_hamming_weight: int,
            max_hamming_weight: int,
            perturb_max_neighbor_hop_range: int,
            perturb_max_hamming_distance: int,
            perturb_max_hamming_weight_change: int,
            device: torch.device,
            dtype: torch.dtype,
            batch_size_tav: int = 1024):
        
        super(BatchedBinaryCodebookManipulator, self).__init__(
            device=device,
            dtype=dtype)
        
        self.n_symbols = 2
        self.code_length = code_length
        self.n_types = n_types
        
        self.min_hamming_weight = min_hamming_weight
        self.max_hamming_weight = max_hamming_weight
        
        self.perturb_max_neighbor_hop_range = perturb_max_neighbor_hop_range
        self.perturb_max_hamming_distance = perturb_max_hamming_distance
        self.perturb_max_hamming_weight_change = perturb_max_hamming_weight_change

        self.device = device
        self.dtype = dtype
        
        self.batch_size_tav = batch_size_tav
        
        # asserts
        assert self.n_types <= self.code_space_size
        assert self.min_hamming_weight >= 0
        assert self.max_hamming_weight <= self.code_length
        assert 1 <= self.perturb_max_hamming_distance <= self.code_length
        assert 0 <= self.perturb_max_hamming_weight_change <= (self.max_hamming_weight - self.min_hamming_weight)
        assert 1 <= self.perturb_max_neighbor_hop_range <= self.n_types
        
        # method cache
        self._method_cache = dict()
        
    @cachedproperty
    def move_types_dict(self) -> Dict[str, int]:
        return {
            'MOVE_SKIP': 0,
            'MOVE_HOPPING': 1,
            'MOVE_HAMMING_DISTANCE': 2,
            'MOVE_HAMMING_WEIGHT': 3}

    @cachedproperty
    def m(self) -> int:
        n_hopping_moves = self.perturb_max_neighbor_hop_range
        n_hamming_distance_moves = self.perturb_max_hamming_distance
        n_hamming_weight_moves = self.perturb_max_hamming_weight_change + 1
        return n_hopping_moves + n_hamming_distance_moves + n_hamming_weight_moves
    
    @cachedproperty
    def move_type_m(self) -> torch.Tensor:
        pert_type_m = torch.zeros((self.m,), dtype=torch.long, device=self.device)
        offset = 0
        pert_type_m[offset:(offset + self.perturb_max_neighbor_hop_range)] = self.MOVE_HOPPING
        offset += self.perturb_max_neighbor_hop_range
        pert_type_m[offset:(offset + self.perturb_max_hamming_distance)] = self.MOVE_HAMMING_DISTANCE
        offset += self.perturb_max_hamming_distance
        pert_type_m[offset:(offset + self.perturb_max_hamming_weight_change + 1)] = self.MOVE_HAMMING_WEIGHT
        return pert_type_m
    
    @cachedproperty
    def move_size_m(self) -> torch.Tensor:
        pert_size_m = torch.zeros((self.m,), dtype=torch.long, device=self.device)
        
        offset = 0
        pert_size_m[offset:(offset + self.perturb_max_neighbor_hop_range)] = 1 + torch.arange(self.perturb_max_neighbor_hop_range)
        offset += self.perturb_max_neighbor_hop_range
        pert_size_m[offset:(offset + self.perturb_max_hamming_distance)] = 1 + torch.arange(self.perturb_max_hamming_distance)
        offset += self.perturb_max_hamming_distance
        pert_size_m[offset:(offset + self.perturb_max_hamming_weight_change + 1)] = torch.arange(self.perturb_max_hamming_weight_change + 1)
        return pert_size_m

    @cachedproperty
    def move_log_tav_m(self) -> torch.Tensor:
        log_target_size_m = torch.zeros((self.m,), dtype=self.dtype, device=self.device)
        
        # finite range uniform hopping
        offset = 0
        log_target_size_m[offset:(offset + self.perturb_max_neighbor_hop_range)] = \
            torch.log(
                torch.arange(
                    1, self.perturb_max_neighbor_hop_range + 1,
                    device=self.device, dtype=self.dtype))
        offset += self.perturb_max_neighbor_hop_range
        
        # hamming distance
        log_target_size_m[offset:(offset + self.perturb_max_hamming_distance)] = \
            self.bcm.cumulative_hamming_distance_tav_d[1:(self.perturb_max_hamming_distance + 1)].log()
        offset += self.perturb_max_hamming_distance
        
        # hamming weight change
        log_target_size_m[offset:(offset + self.perturb_max_hamming_weight_change + 1)] = \
            self.bcm.cumulative_hamming_weight_change_tav_d[0:(self.perturb_max_hamming_weight_change + 1)].log()
        
        # uniform shift for both (choice of the codebook column)
        log_target_size_m += np.log(self.n_types)
        
        return log_target_size_m

    @cachedproperty
    def bool_seq_space_xl(self) -> torch.Tensor:
        return to_torch(
            self.generate_all_codes(
                self.code_length,
                self.min_hamming_weight,
                self.max_hamming_weight),
            device=self.device,
            dtype=torch.bool)
    
    @cachedproperty
    def code_space_size(self) -> int:
        return self.bool_seq_space_xl.shape[0]
    
    @cachedproperty
    def hamming_weight_x(self) -> torch.Tensor:
        return self.bool_seq_space_xl.sum(-1)
    
    @cachedproperty
    def hamming_distance_tav_d(self) -> torch.Tensor:
        """Hamming distance perturbation target accessible volume"""
        sampling_rounds = max(1, int(np.ceil(self.code_space_size / self.batch_size_tav)))
        hamming_distance_tav_d = torch.zeros(
            (self.code_length + 1,), device=self.device, dtype=self.dtype)
        for i_sampling in range(sampling_rounds):
            i_lo = i_sampling * self.batch_size_tav
            i_hi = min((i_sampling + 1) * self.batch_size_tav, self.code_space_size) 
            curr_code_indices_j = torch.arange(i_lo, i_hi)
            hamming_dist_f = \
                (self.bool_seq_space_xl[None, ...] !=
                 self.bool_seq_space_xl[curr_code_indices_j, None, :]).sum(-1).flatten()
            hamming_distance_tav_d += torch.histc(hamming_dist_f, min=0, max=self.code_length, bins=(self.code_length + 1))
        hamming_distance_tav_d[0] = 0. # remove self from target
        return hamming_distance_tav_d / self.code_space_size
    
    @cachedproperty
    def cumulative_hamming_distance_tav_d(self) -> torch.Tensor:
        """Hamming distance perturbation target accessible volume (cumulative)"""
        return torch.cumsum(self.hamming_distance_tav_d, 0)

    @cachedproperty
    def hamming_weight_change_tav_d(self) -> torch.Tensor:
        """Hamming weight perturbation target accessible volume"""
        sampling_rounds = max(1, int(np.ceil(self.code_space_size / self.batch_size_tav)))
        hamming_weight_width = self.max_hamming_weight - self.min_hamming_weight
        hamming_weight_change_tav_d = torch.zeros(
            (hamming_weight_width + 1,), device=self.device, dtype=self.dtype)
        for i_sampling in range(sampling_rounds):
            i_lo = i_sampling * self.batch_size_tav
            i_hi = min((i_sampling + 1) * self.batch_size_tav, self.code_space_size) 
            curr_code_indices_j = torch.arange(i_lo, i_hi)
            hamming_weight_jx = \
                (self.hamming_weight_x[curr_code_indices_j][:, None] -
                 self.hamming_weight_x[None, :]).abs()
            hamming_weight_change_tav_d += torch.histc(
                hamming_weight_jx,
                min=0,
                max=hamming_weight_width,
                bins=hamming_weight_width + 1)
        hamming_weight_change_tav_d[0] -= 1.  # remove self from target
        return hamming_weight_change_tav_d / self.code_space_size
    
    @cachedproperty
    def cumulative_hamming_weight_change_tav_d(self) -> torch.Tensor:
        """Hamming weight perturbation target accessible volume (cumulative)"""
        return torch.cumsum(self.hamming_weight_change_tav_d, 0)            
        
    @staticmethod
    def generate_all_codes(
            code_length: int,
            min_hamming_weight: int,
            max_hamming_weight: int) -> np.ndarray:
        bool_seq_space_xl_np = np.zeros((2 ** code_length, code_length), dtype=bool)
        for i_code in range(2 ** code_length):
            bool_seq_space_xl_np[i_code, :] = [int(b) for b in f'{i_code:0{code_length}b}']

        # filter to codes within the specified Hamming eight bounds
        hamming_weights_x = bool_seq_space_xl_np.sum(-1)
        bool_seq_space_xl_np = bool_seq_space_xl_np[
            (hamming_weights_x >= min_hamming_weight) & (hamming_weights_x <= max_hamming_weight), :]

        return bool_seq_space_xl_np.astype(np.int)
    
    def get_explicit_codebook_btl(
            self,
            indexed_ext_codebook_bx: torch.Tensor) -> torch.Tensor:
        return self.bool_seq_space_xl[indexed_ext_codebook_bx[:, :self.n_types]]
    
    def get_sorted_codebook_btl(
            self,
            codebook_btl: torch.Tensor) -> torch.Tensor:
        """
        .. note::
          the implementation is slow -- this is for visualization purposes only.
        """
        codebook_btl_np = to_np(codebook_btl)
        sorted_codebook_btl_np = np.concatenate(
            [codebook_btl_np[i_b, :, :][:, np.lexsort(codebook_btl_np[i_b, ::-1, :])[::-1]][None, :, :]
             for i_b in range(codebook_btl_np.shape[0])], axis=0)
        return to_torch(sorted_codebook_btl_np, device=self.device, dtype=torch.bool)

    def generate_random_state_batch(
            self,
            batch_size: int) -> torch.Tensor:
        return torch.cat(
            [torch.randperm(self.code_space_size, device=self.device)[None, :]
             for _ in range(batch_size)], 0)

    def perturb(
            self,
            state_bx: torch.Tensor,
            pert_type_bk: torch.Tensor,
            pert_size_bk: torch.Tensor) -> torch.Tensor:
        """This method perturbs a batch of codebooks by performing a series
        of alteration with controlled modification of the Hamming weight.
        
        .. note::
          we do not validate the values of `state_bx`; each codebook is expected to
          be a proper permutation (of all codes)
        
        .. note::
          the first `n_types` entries of each extended codebook is ultimately used
          as the codebook
          
        .. note::
          perturbation types:
            - 0: skip
            - 1: nearest neighbor permutation
            - 2: max Hamming weight change            
        
        :param state_bx: a batch of indexed extended codebooks with
          shape (batch_size, self.code_space_size)
        :param pert_type_bk:
        :param pert_size_bk:
        """
        # basic assert
        assert state_bx.ndim == 2
        batch_size, ext_codebook_length = state_bx.shape
        assert ext_codebook_length == self.code_space_size
        assert pert_type_bk.ndim == 2
        assert pert_size_bk.ndim == 2
        k_max = pert_type_bk.shape[-1]
        assert pert_type_bk.shape == (batch_size, k_max)
        assert pert_size_bk.shape == (batch_size, k_max)

        # aux
        batch_index_range = torch.arange(batch_size, device=self.device)
        
        # clone the original codebook
        new_state_bx = state_bx.clone()
        
        for k in range(k_max):
            
            # choose first in pair
            first_array_indices_b = torch.randint(0, self.n_types, size=[batch_size], device=self.device)
            first_code_indices_b = torch.gather(
                new_state_bx, -1, first_array_indices_b[:, None]).squeeze(-1)
            
            # second in pair (assuming a Hamming distance change move)
            hamming_dist_bx = \
                (self.bool_seq_space_xl[first_code_indices_b][:, None, :]
                 != self.bool_seq_space_xl.expand([batch_size, self.code_space_size, self.code_length])).sum(-1)
            hamming_dist_allowed_second_code_indices_bx = hamming_dist_bx <= pert_size_bk[:, k, None]
            hamming_dist_allowed_second_code_indices_bx[
                batch_index_range, first_code_indices_b] = False  # prevent identity
            hamming_dist_second_code_indices_b = torch.where(
                hamming_dist_allowed_second_code_indices_bx.sum(-1) > 0, 
                torch.argmax(
                    torch.rand([batch_size, self.code_space_size], device=self.device) *
                    hamming_dist_allowed_second_code_indices_bx, -1),
                first_code_indices_b)
            
            # second in pair (assuming a Hamming weight change move)
            hamming_weight_bx = \
                (self.hamming_weight_x[first_code_indices_b][:, None] - self.hamming_weight_x[None, :]).abs()
            hamming_weight_allowed_second_code_indices_bx = hamming_weight_bx <= pert_size_bk[:, k, None]
            hamming_weight_allowed_second_code_indices_bx[
                batch_index_range, first_code_indices_b] = False  # prevent identity
            hamming_weight_second_code_indices_b = torch.where(
                hamming_weight_allowed_second_code_indices_bx.sum(-1) > 0,
                torch.argmax(
                    torch.rand([batch_size, self.code_space_size], device=self.device) *
                    hamming_weight_allowed_second_code_indices_bx, -1),
                first_code_indices_b)

            # second in pair (assuming a nearest neighbor permutation move)
            hop_size_b = torch.ceil(
                pert_size_bk[:, k].type(self.dtype) *
                torch.rand((batch_size,), device=self.device, dtype=self.dtype)).type(torch.long)
            hopping_second_array_indices_b = torch.clamp(
                first_array_indices_b + hop_size_b,
                0, self.code_space_size - 1)
            hopping_second_code_indices_b = new_state_bx[
                batch_index_range, hopping_second_array_indices_b]
            
            # choose the move according to pert_type_bk
            second_code_indices_b = first_code_indices_b.clone()  # initialize with `skip`
            second_code_indices_b = torch.where(
                pert_type_bk[:, k] == self.MOVE_HOPPING,
                hopping_second_code_indices_b,
                second_code_indices_b)
            second_code_indices_b = torch.where(
                pert_type_bk[:, k] == self.MOVE_HAMMING_DISTANCE,
                hamming_dist_second_code_indices_b,
                second_code_indices_b)
            second_code_indices_b = torch.where(
                pert_type_bk[:, k] == self.MOVE_HAMMING_WEIGHT,
                hamming_weight_second_code_indices_b,
                second_code_indices_b)

            # apply the perturbation
            pert_bx = torch.arange(self.code_space_size, device=self.device)[None, :]\
                .expand([batch_size, self.code_space_size])\
                .contiguous()
            pert_bx[batch_index_range, first_code_indices_b] = second_code_indices_b
            pert_bx[batch_index_range, second_code_indices_b] = first_code_indices_b
            new_state_bx = torch.gather(pert_bx, -1, new_state_bx)
            
        return new_state_bx


class NeuralMCMCAcceptanceRatePredictor(MCMCAcceptanceRatePredictor):
    """Predicts MCMC acceptance rate given temperature, initial energy, and path.
    
    .. note:
      it is assumed that the # of moves is much smaller than the state size, so
      that the moves are independent ("non-interacting"). therefore, we can model
      the acceptance rate of a path as a permutation-invariant function of the
      moves (i.e. unordered).
    """
    
    def __init__(
            self,
            n_atomic_moves: int,
            device: torch.device,
            dtype: torch.dtype):
        super(NeuralMCMCAcceptanceRatePredictor, self).__init__(
            device=device,
            dtype=dtype)
        
        self.n_atomic_moves = n_atomic_moves
        
        self.move_mapper = generate_dense_nnet(
            [(n_atomic_moves + 2, 20),
             'selu',
             (20, 10),
             'selu',
             (10, 10)]
        ).to(device)
        
        self.path_reducer = generate_dense_nnet(
            [(10, 10),
             'selu',
             (10, 10),
             'selu',
             (10, 1)]
        ).to(device)
        
    def forward(
            self,
            pert_nbkm: torch.Tensor,
            log_temperature_b: torch.Tensor,
            initial_energy_b: torch.Tensor) -> torch.Tensor:
        
        # add initial energy and temperature as features
        input_nbkf = torch.cat(
            (pert_nbkm,
             log_temperature_b[None, :, None, None].expand(pert_nbkm.shape[:-1] + (1,)),
             initial_energy_b[None, :, None, None].expand(pert_nbkm.shape[:-1] + (1,))),
            dim=-1) 
        
        # map
        mapped_pert_nbkq = self.move_mapper(input_nbkf)
        
        # reduce
        reduced_pert_nb = self.path_reducer(mapped_pert_nbkq.sum(-2)).squeeze(-1)
        
        return reduced_pert_nb


class BatchedBinaryCodebookEnergyCalculator(BatchedStateEnergyCalculator):
    
    def __init__(
            self,
            experiment_spec: ExperimentSpecification,
            bac_channel_spec: BinaryChannelSpecification,
            metric_type: str,
            batched_binary_codebook_manipulator: BatchedBinaryCodebookManipulator,
            split_size: int,
            device: torch.device,
            dtype: torch.dtype,
            **kwargs):
                
        self.metric_type = metric_type
        self.experiment_spec = experiment_spec
        self.bac_channel_spec = bac_channel_spec
        self.batched_binary_codebook_manipulator = batched_binary_codebook_manipulator
        
        self.device = device
        self.dtype = dtype
        
        self.pi_t = to_torch(experiment_spec.pi_t, self.device, self.dtype)
        self.mean_reduce = lambda metric_bt: metric_bt.mean(-1)
        self.complement = lambda metric_b: (1. - metric_b)
        self.state_bx_to_codebook_btl = self.batched_binary_codebook_manipulator.get_explicit_codebook_btl
        
        # set the state energy calculator
        optional_fdr_quantile = self.parse_quantile_based_metric_type_str('fdr', metric_type)
        optional_tpr_quantile = self.parse_quantile_based_metric_type_str('tpr', metric_type)
        f1_reject_auc_metrics_dict_calculator = lambda state_bx: \
            calculate_bac_f1_reject_auc_metric_dict(
                codebook_btl=self.state_bx_to_codebook_btl(state_bx),
                pi_t=self.pi_t,
                bac_model=self.bac_channel_spec.channel_model,
                delta_q_max=delta_q_max,
                n_map_reject_thresholds=n_map_reject_thresholds,
                max_reject_ratio=max_reject_ratio,
                split_size=split_size,
                device=self.device,
                dtype=dtype)
        basic_metrics_dict_calculator = lambda state_bx: \
            calculate_bac_standard_metric_dict(
                codebook_btl=self.state_bx_to_codebook_btl(state_bx),
                pi_t=self.pi_t,
                bac_model=self.bac_channel_spec.channel_model,
                split_size=split_size)

        
        if metric_type == 'f1_reject_auc':
            assert 'delta_q_max' in kwargs
            assert 'n_map_reject_thresholds' in kwargs
            assert 'max_reject_ratio' in kwargs
            delta_q_max = kwargs['delta_q_max']
            n_map_reject_thresholds = kwargs['n_map_reject_thresholds']
            max_reject_ratio = kwargs['max_reject_ratio']
            self._calculate_state_metrics_dict = f1_reject_auc_metrics_dict_calculator
            self._calculate_state_energy = lambda state_bx: \
                self.mean_reduce(
                    self.complement(
                        self._calculate_state_metrics_dict(state_bx)['normalized_auc_f_1_rej_bt']))
            
        elif metric_type == 'fdr':
            self._calculate_state_metrics_dict = basic_metrics_dict_calculator
            self._calculate_state_energy = lambda state_bx: \
                self.mean_reduce(
                    self._calculate_state_metrics_dict(state_bx)['fdr_bt'])
            
        elif metric_type == 'tpr':          
            self._calculate_state_metrics_dict = basic_metrics_dict_calculator
            self._calculate_state_energy = lambda state_bx: \
                self.mean_reduce(
                    self.complement(
                        self._calculate_state_metrics_dict(state_bx)['tpr_bt']))
            
        elif optional_fdr_quantile is not None:
            self._calculate_state_metrics_dict = basic_metrics_dict_calculator
            self._calculate_state_energy = lambda state_bx: \
                self.mean_reduce(
                    self.get_top_quantile(
                        quantile=optional_fdr_quantile,
                        metric_bt=self._calculate_state_metrics_dict(state_bx)['fdr_bt']))
            
        elif optional_tpr_quantile is not None:
            self._calculate_state_metrics_dict = basic_metrics_dict_calculator
            self._calculate_state_energy = lambda state_bx: \
                self.mean_reduce(
                    self.get_top_quantile(
                        quantile=optional_fdr_quantile,
                        metric_bt=self.complement(self._calculate_state_metrics_dict(state_bx)['tpr_bt'])))
        else:
            raise ValueError(
                'Unknown metric type -- allowed values: '
                'f1_reject_auc, fdr, tpr, fdr[<quantile>], tpr[<quantile>]')
    
    def get_top_quantile(self, quantile: float, metric_bt: torch.Tensor) -> torch.Tensor:
        n_types = metric_bt.size(1)
        n_keep = math.ceil(n_types * quantile)
        sorted_metric_bt = torch.sort(metric_bt, dim=-1).values
        return sorted_metric_bt[:, -n_keep:]
        
    def parse_quantile_based_metric_type_str(
            self,
            base_metric_type: str,
            metric_type_string: str) -> Optional[float]:
        if metric_type_string[:4] == (base_metric_type + '[') and metric_type_string[-1] == ']':
            try:
                i_begin = metric_type_string.find('[') + 1
                i_end = metric_type_string.find(']')
                assert i_end > i_begin
                quantile = float(metric_type_string[i_begin:i_end])
                assert 0. < quantile < 1.
                return quantile
            except:
                return None
        return None

    def calculate_state_energy(self, state_bx: torch.Tensor) -> torch.Tensor:
        return self._calculate_state_energy(state_bx)
    
    def calculate_state_metrics_dict(self, state_bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._calculate_state_metrics_dict(state_bx)

    
class SimulatedAnnealingBinaryAsymmetricChannel:
    
    @staticmethod
    def init_from_yaml(
            input_yaml_file: str,
            logger: Optional[Callable] = None) -> 'SimulatedAnnealingBinaryAsymmetricChannel':
        try:
            with open(input_yaml_file, 'r') as f:
                params = yaml.load(f)
        except IOError:
            raise RuntimeError(f"Error loading input YAML file {input_yaml_file}!")
        
        return SimulatedAnnealingBinaryAsymmetricChannel(
            params=params,
            logger=logger)

    def __init__(
            self,
            params: CommentedMap,
            logger: Optional[Callable] = None):

        if logger is None:
            logger = log_info
        self.log_info = logger
        
        self.log_info("Simulated annealing starting ...")

        self.params = params
        self.device = consts.DEFAULT_DEVICE_EVAL
        self.dtype = consts.DEFAULT_DTYPE

        # fetch all BAC channel models
        all_bac_channel_model_specs_dict: Dict[str, ChannelModelSpecification] = {}
        for k, v in experiments.__dict__.items():
            if isinstance(v, ChannelModelSpecification):
                if isinstance(v.channel_model, BinaryAsymmetricChannelModel):
                    all_bac_channel_model_specs_dict[k] = v

        assert params['channel_model'] in all_bac_channel_model_specs_dict, \
            f"Bad channel model -- allowed values are: {', '.join(list(all_bac_channel_model_specs_dict.keys()))}"
        self.channel_spec = all_bac_channel_model_specs_dict[params['channel_model']]

        # generate experiment spec
        self.experiment_spec = experiments.generate_experiment_spec(
            name_prefix=params['experiment_prefix'],
            min_symbol_weight_s=[
                params['min_hamming_weight'],
                params['min_hamming_weight']],
            max_symbol_weight_s=[
                params['max_hamming_weight'],
                params['max_hamming_weight']],
            n_symbols=2,
            code_length=params['code_length'],
            n_types=params['n_types'],
            source_nonuniformity=params['source_nonuniformity'])

        # generate problem spec
        self.problem_spec = experiments.SingleEntityCodingProblemSpecification(
            experiment_spec=self.experiment_spec,
            channel_spec=self.channel_spec)

        # generate output path
        self.output_root = params['output_root']
        self.output_prefix = self.problem_spec.name + '__' + params['metric_type']
        self.output_path = os.path.join(self.output_root, self.output_prefix)

        # log problem specification
        self.log_info(pprint.pformat(self.problem_spec))

        # instantiate the state manipulator
        self.batched_binary_codebook_manipulator = BatchedBinaryCodebookManipulator(
            code_length=self.experiment_spec.code_length,
            n_types=self.experiment_spec.n_types,
            min_hamming_weight=int(self.experiment_spec.min_symbol_weight_s[1]),
            max_hamming_weight=int(self.experiment_spec.max_symbol_weight_s[1]),
            perturb_max_neighbor_hop_range=params['perturb_max_neighbor_hop_range'],
            perturb_max_hamming_distance=params['perturb_max_hamming_distance'],
            perturb_max_hamming_weight_change=params['perturb_max_hamming_weight_change'],
            device=self.device,
            dtype=self.dtype)

        # instantiate the state energy calculator
        self.batched_binary_codebook_energy_calculator = BatchedBinaryCodebookEnergyCalculator(
            experiment_spec=self.experiment_spec,
            bac_channel_spec=self.channel_spec,
            metric_type=params['metric_type'],
            batched_binary_codebook_manipulator=self.batched_binary_codebook_manipulator,
            split_size=params['eval_split_size'],
            device=self.device,
            dtype=self.dtype)

        # instantiate MCMC acceptance rate predictor
        self.neural_mcmc_acceptance_rate_predictor = NeuralMCMCAcceptanceRatePredictor(
            n_atomic_moves=self.batched_binary_codebook_manipulator.m,
            device=self.device,
            dtype=self.dtype)

        # load checkpoint
        self.checkpoint_path = os.path.join(self.output_root, params['checkpoint_file'])
        checkpoint_load_success = True
        if not params['ignore_checkpoint']:
            if os.path.exists(self.checkpoint_path):    
                self.log_info(f"Found checkpoint at: {self.checkpoint_path}")
                try:
                    self.log_info("Extracting the checkpoint ...")
                    # extract the checkpoint tarball
                    with tarfile.open(self.checkpoint_path) as tar:
                        tar.extractall(self.output_root)

                    # assert the validity of checkpoint
                    if not os.path.exists(self.output_path):
                        self.log_info(f"Invalid checkpoint -- {self.output_path} does not exist in the tarball!")
                        checkpoint_load_success = False
                    if not os.path.exists(os.path.join(self.output_path, "params.yaml")):
                        self.log_info("params.yaml is missing from the tarball!")
                        checkpoint_load_success = False
                    if not os.path.exists(os.path.join(self.output_path, "latest_state.pkl")):
                        self.log_info("latest_state.pkl is missing from the tarball!")
                        checkpoint_load_success = False

                    # load the the extracted checkpoint
                    if checkpoint_load_success:
                        self.log_info("Loading params.yaml from the checkpoint file ...")
                        with open(os.path.join(self.output_path, "params.yaml")) as params_yaml_file:
                            loaded_params = yaml.load(params_yaml_file)
                        self.log_info("Loading latest_state.pkl from the checkpoint file ...")
                        loaded_state_dict = torch.load(os.path.join(self.output_path, "latest_state.pkl"))
                except:
                    self.log_info("Error occurred while loading checkpoint -- possibly corrupt?")
                    checkpoint_load_success = False
            else:
                checkpoint_load_success = False
        else:
            checkpoint_load_success = False

        # if the checkpoint is extract, the output path exists; otherwise, we need to make it 
        os.makedirs(self.output_path, exist_ok=True)

        if checkpoint_load_success:
            self.log_info("Using the previously calculated energy spread from the checkpoint ...")
            energy_moments_dict = dict()
            energy_moments_dict['energy_loc'] = loaded_params['rand_state_energy_loc']
            energy_moments_dict['energy_scale'] = loaded_params['rand_state_energy_scale']
        else:
            # calculate the energy spread of random codebooks
            self.log_info("Determining energy spread ...")
            energy_moments_dict = estimate_energy_scale(
                n_rounds=params['energy_spread_estimation_n_rounds'],
                batch_size=params['energy_spread_estimation_batch_size'],
                batched_state_energy_calculator=self.batched_binary_codebook_energy_calculator,
                batched_state_manipulator=self.batched_binary_codebook_manipulator,
                device=self.device,
                dtype=self.dtype)

        params['rand_state_energy_loc'] = energy_moments_dict['energy_loc']
        params['rand_state_energy_scale'] = energy_moments_dict['energy_scale']

        # save parameters to output path
        with open(os.path.join(self.output_path, "params.yaml"), 'w') as f:
            yaml.dump(params, f)

        self.log_info("Instantiating simulated annealing workspace ...")
        self.ptpsa = PyTorchBatchedSimulatedAnnealing(
            batched_state_energy_calculator=self.batched_binary_codebook_energy_calculator,
            batched_state_manipulator=self.batched_binary_codebook_manipulator,
            mcmc_acceptance_rate_predictor=self.neural_mcmc_acceptance_rate_predictor,
            device=self.device,
            dtype=self.dtype,
            **params)

        if checkpoint_load_success:
            self.log_info("Setting the state_dict from the checkpoint ...")
            self.ptpsa.load_state_dict(loaded_state_dict)

    def save_results(self, output_tar_gz_prefix: str):
        # save ptpsa state
        torch.save(
            self.ptpsa.state_dict(),
            os.path.join(self.output_path, 'latest_state.pkl'))

        # make plots
        self.log_info("Making plots ...")
        if self.params['make_plots']:

            plot_sa_trajectory(
                ptpsa=self.ptpsa,
                output_path=self.output_path,
                output_prefix='latest_trajectory',
                show=False)

            plot_sa_codebook(
                ptpsa=self.ptpsa,
                codebook_bx=self.ptpsa.top_state_kx,
                energy_b=self.ptpsa.top_energy_k,
                idx=0,
                batch_name='Top states',
                output_path=self.output_path,
                output_prefix='latest_codebook',
                show=False)

            plot_sa_resampling_buffer(
                ptpsa=self.ptpsa,
                output_path=self.output_path,
                output_prefix='resampling_buffer',
                show=False)

        # make a .tar.gz file
        self.log_info("Compressing the output ...")
        tmp_output_path = os.path.join(self.output_root, f'_{output_tar_gz_prefix}.tar.gz')
        final_output_path = os.path.join(self.output_root, f'{output_tar_gz_prefix}.tar.gz')
        with tarfile.open(tmp_output_path, 'w:gz') as tar:
            tar.add(self.output_path, arcname=self.output_prefix)
        os.replace(tmp_output_path, final_output_path)

    def run(self):
        self.log_info("Starting the simulation ...")
        torch.cuda.empty_cache()

        # logging parameters
        log_frequency = self.params['log_frequency']
        checkpoint_interval_seconds = self.params['checkpoint_interval_seconds']
        small_col = self.params['log_column_width_small']
        large_col = self.params['log_column_width_large']
        sig_digits = self.params['log_sig_digits']
        log_energy_scale = self.params['log_energy_scale']

        header_string = \
            f"{'i_iter'.ljust(small_col)}" \
            f"{'temp'.ljust(small_col)}" \
            f"{'energies'.ljust(large_col)}" \
            f"{'acc emp'.ljust(small_col)}" \
            f"{'acc pred'.ljust(small_col)}" \
            f"{'acc penalty'.ljust(small_col)}" \
            f"{'lowest'.ljust(small_col)}" \

        self.log_info(f"Logging energy and temperature scale: {log_energy_scale:.3f}")
        self.log_info(header_string)
        self.log_info('=' * len(header_string))

        exit_code = SimulatedAnnealingExitCode.CONTINUE
        t_last_checkpoint = time.perf_counter()
        while exit_code == SimulatedAnnealingExitCode.CONTINUE:
            current_time = time.perf_counter()

            # step
            exit_code = self.ptpsa.step()

            # log
            if self.ptpsa.i_iter > 1 and self.ptpsa.i_iter % log_frequency == 1:

                temperature = log_energy_scale / self.ptpsa.beta
                min_energy = log_energy_scale * self.ptpsa.energy_b.min().item()
                max_energy = log_energy_scale * self.ptpsa.energy_b.max().item()
                mean_energy = log_energy_scale * self.ptpsa.energy_b.mean().item()
                prob_mcmc_acc_emp_unbiased = log_energy_scale * self.ptpsa.prob_mcmc_acc_emp_unbiased
                prob_mcmc_acc_pred_unbiased = log_energy_scale * self.ptpsa.prob_mcmc_acc_pred_unbiased
                prob_mcmc_acc_penalty = self.ptpsa.log_lambda_mcmc_acc_rate.exp().item()
                lowest_energy = log_energy_scale * self.ptpsa.top_energy

                log_string = \
                    f"{str(self.ptpsa.i_iter).ljust(small_col)}" + \
                    f"{temperature:.{sig_digits}f}".ljust(small_col) + \
                    f"{mean_energy:.{sig_digits}f} ({min_energy:.{sig_digits}f}, {max_energy:.{sig_digits}f})".ljust(large_col) + \
                    f"{prob_mcmc_acc_emp_unbiased:.{sig_digits}f}".ljust(small_col) + \
                    f"{prob_mcmc_acc_pred_unbiased:.{sig_digits}f}".ljust(small_col) + \
                    f"{prob_mcmc_acc_penalty:.{sig_digits}f}".ljust(small_col) + \
                    f"{lowest_energy:.{sig_digits}f}".ljust(small_col)

                self.log_info(log_string)

            # checkpoint
            if (current_time - t_last_checkpoint) > checkpoint_interval_seconds:
                t_last_checkpoint = current_time
                self.log_info("Checkpointing the simulation state ...")
                self.save_results(output_tar_gz_prefix='checkpoint')
        
        # end with a final checkpointing
        self.log_info("Saving the final simulation state ...")
        self.save_results(output_tar_gz_prefix='final_state')
        self.log_info(f"Simulation concluded -- exit code: {exit_code}")
