import numpy as np
import torch
from typing import Union, NamedTuple, Optional, Dict, List, Tuple, Set
from boltons.cacheutils import cachedmethod
from abc import abstractmethod

from mighty_codes.torch_utils import \
    to_torch, \
    to_np, \
    to_one_hot_encoded, \
    split_tensors

from mighty_codes.metric_utils import \
    get_confusion_matrix_from_indices, \
    get_log_prob_map_thresholds_q


class SingleEntityChannelModel:
    def __init__(self, n_symbols: int):
        self.n_symbols = n_symbols

    @abstractmethod
    def get_weighted_confusion_matrix(
            self,
            codebook_btls: torch.Tensor,
            pi_bt: torch.Tensor,
            decoder_type: str,
            **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @abstractmethod
    def to(self, device: torch.device) -> 'SingleEntityChannelModel':
        raise NotImplementedError

    @abstractmethod
    def type(self, dtype: torch.dtype) -> 'SingleEntityChannelModel':
        raise NotImplementedError

    
class GaussianChannelModel(SingleEntityChannelModel):
    def __init__(
            self,
            n_symbols: int,
            n_readout: int,
            loc_sr: torch.Tensor,
            scale_sr: torch.Tensor,
            device: torch.device,
            dtype: torch.dtype):
        super(GaussianChannelModel, self).__init__(n_symbols=n_symbols)

        self.n_readout = n_readout

        self.loc_sr = to_torch(loc_sr, device=device, dtype=dtype)
        self.scale_sr = to_torch(scale_sr, device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        assert loc_sr.shape == (n_symbols, n_readout)
        assert scale_sr.shape == (n_symbols, n_readout)

    def to(self, device: torch.device) -> 'GaussianChannelModel':
        return GaussianChannelModel(
            n_symbols=self.n_symbols,
            n_readout=self.n_readout,
            loc_sr=self.loc_sr.to(device),
            scale_sr=self.scale_sr.to(device),
            device=device,
            dtype=self.dtype)

    def type(self, dtype: torch.dtype) -> 'GaussianChannelModel':
        return GaussianChannelModel(
            n_symbols=self.n_symbols,
            n_readout=self.n_readout,
            loc_sr=self.loc_sr.type(dtype),
            scale_sr=self.scale_sr.type(dtype),
            device=self.device,
            dtype=dtype)

    def rsample_flat(
            self,
            n_samples_per_type: int,
            codebook_tls: torch.Tensor) -> torch.Tensor:
        """
        :param n_samples_per_type: number of samples per type
        :param codebook_tls: soft codebook (>=0, must sum to 1 along the last dimension)
        
        .. note:
          the shape of codebook_tls is (n_types, code_length, n_symbols)
        
        .. return:
          a torch tensor with shape (n_samples, code_length, n_readout)
        """
        
        # asserts
        assert codebook_tls.ndim == 3
        n_types, code_length, n_symbols = codebook_tls.shape
        assert n_symbols == self.n_symbols
        device = codebook_tls.device
        
        # the effective readout loc and scale
        effective_loc_tlr = torch.matmul(codebook_tls, self.loc_sr)
        effective_scale_tlr = torch.matmul(codebook_tls, self.scale_sr)
        
        # sample types
        source_type_index_n = torch.repeat_interleave(
            torch.arange(0, n_types, dtype=torch.long, device=device), n_samples_per_type)
        
        # sample readouts
        rsampled_readout_nlr = torch.distributions.Normal(
            loc=effective_loc_tlr[source_type_index_n, ...],
            scale=effective_scale_tlr[source_type_index_n, ...]).rsample()
        
        return source_type_index_n, rsampled_readout_nlr

    
    def calculate_log_posterior_probs(
            self,
            pi_t: torch.Tensor,
            codebook_tls: torch.Tensor,
            readout_nlr: torch.Tensor) -> torch.Tensor:
        """
        :param pi_t: prior
        :param codebook_tls: soft codebook (>=0, must sum to 1 along the last dimension)
        :param readout_nlr: readout tensor to be decoded
        
        .. note:
          the shape of codebook_tls is (n_types, code_length, n_symbols)
        
        .. note:
          the shape of readout_nlr is (n_samples, code_length, n_readout)
          
        .. return:
          a log probabilty tensor with shape (n_samples, n_types); sums to be 1 along the
          last dimension once exponentiated
        """
        
        # asserts
        n_types = pi_t.numel()
        assert pi_t.shape == (n_types,)
        assert codebook_tls.ndim == 3
        n_types, code_length, n_symbols = codebook_tls.shape
        assert n_symbols == self.n_symbols

        # the effective readout loc and scale
        effective_loc_tlr = torch.matmul(codebook_tls, self.loc_sr)
        effective_scale_tlr = torch.matmul(codebook_tls, self.scale_sr)

        # calcualte the log likelihoods
        log_like_nt = torch.distributions.Normal(
            loc=effective_loc_tlr[None, ...],
            scale=effective_scale_tlr[None, ...]).log_prob(readout_nlr[:, None, :, :]).sum(dim=(-1, -2))
        
        # calculate posterior
        log_pi_t = pi_t.log()
        log_posterior_nt = log_like_nt + log_pi_t[None, :]
        log_posterior_nt = log_posterior_nt - torch.logsumexp(log_posterior_nt, dim=-1, keepdim=True)
        
        return log_posterior_nt
    
    def batched_rsample_flat(
            self,
            n_samples_per_type: int,
            codebook_btls: torch.Tensor) -> torch.Tensor:
        """
        :param n_samples_per_type: number of samples per type
        :param codebook_btls: soft codebook (>=0, must sum to 1 along the last dimension)
        
        .. note:
          the shape of codebook_btls is (batch_size, n_types, code_length, n_symbols)
        
        .. return:
          TBW
        """
        
        # asserts
        assert codebook_btls.ndim == 4
        batch_size, n_types, code_length, n_symbols = codebook_btls.shape
        assert n_symbols == self.n_symbols
        
        # the effective readout loc and scale
        effective_loc_btlr = torch.einsum('btls,sr->btlr', codebook_btls, self.loc_sr)
        effective_scale_btlr = torch.einsum('btls,sr->btlr', codebook_btls, self.scale_sr)
        
        # sample types
        source_type_index_n = torch.repeat_interleave(
            torch.arange(0, n_types, dtype=torch.long, device=codebook_btls.device),
            n_samples_per_type)
        
        # sample readouts
        rsampled_readout_bnlr = torch.distributions.Normal(
            loc=effective_loc_btlr[:, source_type_index_n, :, :],
            scale=effective_scale_btlr[:, source_type_index_n, :, :]).rsample()
        
        return source_type_index_n, rsampled_readout_bnlr

    def batched_calculate_log_posterior_probs(
            self,
            pi_bt: torch.Tensor,
            codebook_btls: torch.Tensor,
            readout_bnlr: torch.Tensor) -> torch.Tensor:
        """
        :param pi_bt: prior
        :param codebook_btls: soft codebook (>=0, must sum to 1 along the last dimension)
        :param readout_bnlr: readout tensor to be decoded
        
        .. note:
          the shape of codebook_btls is (batch_size, n_types, code_length, n_symbols)
        
        .. note:
          the shape of readout_bnlr is (batch_size, n_samples, code_length, n_readout)
          
        .. return:
          a log probabilty tensor with shape (batch_size, n_samples, n_types); sums to be 1 along the
          last dimension once exponentiated
        """
        
        # asserts
        assert pi_bt.ndim == 2
        assert codebook_btls.ndim == 4
        assert readout_bnlr.ndim == 4
        
        batch_size, n_types, code_length, n_symbols = codebook_btls.shape
        n_samples = readout_bnlr.shape[1]
        assert n_symbols == self.n_symbols        
        assert pi_bt.shape == (batch_size, n_types)
        assert readout_bnlr.shape == (batch_size, n_samples, code_length, self.n_readout)

        # the effective readout loc and scale
        effective_loc_btlr = torch.einsum('btls,sr->btlr', codebook_btls, self.loc_sr)
        effective_scale_btlr = torch.einsum('btls,sr->btlr', codebook_btls, self.scale_sr)

        # calcualte the log likelihoods
        log_like_bnt = torch.distributions.Normal(
            loc=effective_loc_btlr[:, None, :, :, :],
            scale=effective_scale_btlr[:, None, :, :, :]).log_prob(readout_bnlr[:, :, None, :, :]).sum(dim=(-1, -2))
        
        # calculate posterior
        log_pi_bt = pi_bt.log()
        log_posterior_bnt = log_like_bnt + log_pi_bt[:, None, :]
        log_posterior_bnt = log_posterior_bnt - torch.logsumexp(log_posterior_bnt, dim=-1, keepdim=True)

        return log_posterior_bnt
    
    def estimate_bac_channel_parameters(
            self,
            n_samples_per_symbol: int):
        """Estimates the approximately equivanelt BAC channel."""

        source_symbols_n = []
        mle_symbol_n = []
        
        for s in range(self.n_symbols):
            # sample from the symbol
            readout_nr = torch.distributions.Normal(
                loc=self.loc_sr[s, :],
                scale=self.scale_sr[s, :]).sample([n_samples_per_symbol])
            # mle decode
            log_like_ns = torch.distributions.Normal(
                loc=self.loc_sr[None, ...],
                scale=self.scale_sr[None, ...]).log_prob(readout_nr[:, None, :]).sum(dim=-1)
            source_symbols_n += ([s] * n_samples_per_symbol)
            mle_symbol_n += torch.argmax(log_like_ns, axis=-1).cpu().numpy().tolist()
        
        confusion_matrix_ss = get_confusion_matrix_from_indices(
            source_indices=source_symbols_n,
            target_indices=mle_symbol_n,
            n_classes=self.n_symbols) / n_samples_per_symbol
        
        return confusion_matrix_ss

    def estimate_weighted_confusion_matrix_soft_mc(
            self,
            codebook_btls: torch.Tensor,
            pi_bt: torch.Tensor,
            n_samples_per_type: int,
            max_n_samples_per_type_per_sampling_round: int) -> torch.Tensor:
        """Estimates the weighted confusion matrix for a given batch of codebooks
        via Monte-Carlo sampling for a Gaussian channel. The decoder is assumed to
        be soft (e.g. stochastic sampling from the posterior).

        .. note::
          The output is differentable with respect to the codebook (e.g. useful
          if the codebook is obtained from a Gumble-Softmax process).
        """
        # shape
        batch_size, n_types, code_length, n_symbols = codebook_btls.shape
        device, dtype = codebook_btls.device, codebook_btls.dtype

        # batched confusion matrix container
        weighted_confusion_matrix_btt = torch.zeros(
            (batch_size, n_types, n_types),
            device=device, dtype=dtype)

        i_samples_per_type = 0
        while i_samples_per_type < n_samples_per_type:

            # how many samples per type to draw?
            c_samples_per_type = min(
                max_n_samples_per_type_per_sampling_round,
                n_samples_per_type - i_samples_per_type)

            # get a differentiable readout sample
            source_type_index_n, rsampled_readout_bnlr = self.batched_rsample_flat(
                n_samples_per_type=c_samples_per_type,
                codebook_btls=codebook_btls)

            # calculate posterior probabilities
            posterior_probs_bnt = self.batched_calculate_log_posterior_probs(
                pi_bt=pi_bt,
                codebook_btls=codebook_btls,
                readout_bnlr=rsampled_readout_bnlr).exp()

            # calculate the confusion matrix
            source_weight_bnt = (pi_bt[:, None, :] / n_samples_per_type) * to_one_hot_encoded(
                type_indices_n=source_type_index_n,
                n_types=n_types).type(self.dtype)

            # update confusion matrix
            weighted_confusion_matrix_btt += torch.einsum(
                'bnt,bnr->btr', source_weight_bnt, posterior_probs_bnt)

            i_samples_per_type += c_samples_per_type

        return weighted_confusion_matrix_btt
    
    def estimate_weighted_confusion_matrix_map_reject_mc(
            self,
            codebook_btls: torch.Tensor,
            pi_bt: torch.Tensor,
            n_samples_per_type: int,
            max_n_samples_per_type_per_sampling_round: int,
            loq_prob_map_thresholds_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimates the weighted confusion matrix for a given batch of codebooks
        via Monte-Carlo sampling for a Gaussian channel. The decoder is assumed to
        be hard (MAP) with a reject option based on posterior concentration.
        
        .. note::
          The output is differentable with respect to the codebook (e.g. useful
          if the codebook is obtained from a Gumble-Softmax process).

        :param model: an instance of `GaussianChannelModel`
        :param codebook_btls: a batch of codebooks
        :param pi_bt: a batch of source priors
        :param n_samples_per_type: total number of MC samples per type
        :param max_n_samples_per_type_per_sampling_round: number of MC samples per type per sampling round
        :param loq_prob_map_thresholds_q: rejection thresholds (minimum log posterior prob concentration)
        """
        # asserts
        assert torch.all(loq_prob_map_thresholds_q <= 0.)
        assert loq_prob_map_thresholds_q.ndim == 1

        # shape
        batch_size, n_types, code_length, n_symbols = codebook_btls.shape
        n_thresholds = loq_prob_map_thresholds_q.shape[-1]
        assert n_thresholds > 0
        device, dtype = codebook_btls.device, codebook_btls.dtype

        # batched confusion matrix container with shape:
        #   (batch_size, n_thresholds, SOURCE = n_types, TARGET = n_types + 1)
        # 
        # ..note::
        #   the last TARGET is the reject option
        weighted_confusion_matrix_bqtu = torch.zeros(
            (batch_size, n_thresholds, n_types, n_types + 1),
            device=device, dtype=dtype)

        # keep track of the number of all + rejected samples of each type
        num_all_bt = torch.zeros(
            (batch_size, n_types),
            device=device, dtype=torch.long)
        num_rej_bqt = torch.zeros(
            (batch_size, n_thresholds, n_types),
            device=device, dtype=torch.long)

        # used to calculate collaped (MAP) posterior
        eye_uu = torch.eye(n_types + 1, device=device, dtype=dtype)

        i_samples_per_type = 0
        while i_samples_per_type < n_samples_per_type:

            # how many samples per type to draw?
            c_samples_per_type = min(
                max_n_samples_per_type_per_sampling_round,
                n_samples_per_type - i_samples_per_type)

            # get a differentiable readout sample
            source_type_index_n, rsampled_readout_bnlr = self.batched_rsample_flat(
                n_samples_per_type=c_samples_per_type,
                codebook_btls=codebook_btls)

            # calculate posterior probabilities
            log_posterior_probs_bnt = self.batched_calculate_log_posterior_probs(
                pi_bt=pi_bt,
                codebook_btls=codebook_btls,
                readout_bnlr=rsampled_readout_bnlr)

            # source weight 
            source_weight_bnt = (pi_bt[:, None, :] / n_samples_per_type) * to_one_hot_encoded(
                type_indices_n=source_type_index_n,
                n_types=n_types).type(self.dtype)

            # MAP decoding call
            map_bn = log_posterior_probs_bnt.argmax(-1)

            # posterior concentration
            log_prob_map_bn = torch.gather(log_posterior_probs_bnt, -1, map_bn.unsqueeze(-1)).squeeze(-1)

            # calculate the confusion matrix
            map_reject_bnq = map_bn.unsqueeze(-1).expand(map_bn.shape + (n_thresholds,)).contiguous()
            rej_mask_bnq = log_prob_map_bn.unsqueeze(-1) < loq_prob_map_thresholds_q
            map_reject_bnq[rej_mask_bnq] = n_types
            collasped_posterior_probs_bnqu = eye_uu[map_reject_bnq, :]

            # update confusion matrix
            weighted_confusion_matrix_bqtu += torch.einsum(
                'bnt,bnqu->bqtu', source_weight_bnt, collasped_posterior_probs_bnqu)

            # update rejected counts
            num_rej_bqt.index_add_(
                -1, source_type_index_n,
                rej_mask_bnq.permute(0, 2, 1).long())

            # update all counts
            num_all_bt.index_add_(
                -1, source_type_index_n,
                torch.ones_like(source_type_index_n).expand((batch_size, -1)))

            i_samples_per_type += c_samples_per_type

        return {
            'weighted_confusion_matrix_bqtu': weighted_confusion_matrix_bqtu,
            'num_all_bt': num_all_bt,
            'num_rej_bqt': num_rej_bqt
        }

    def get_weighted_confusion_matrix(
            self,
            codebook_btls: torch.Tensor,
            pi_bt: torch.Tensor,
            decoder_type: str,
            **kwargs) -> Dict[str, torch.Tensor]:
        
        # basic asserts
        assert isinstance(codebook_btls, torch.Tensor)
        assert codebook_btls.ndim == 4
        batch_size, n_types, code_length, n_symbols = codebook_btls.shape
        assert n_symbols == self.n_symbols
        assert isinstance(pi_bt, torch.Tensor)
        assert pi_bt.ndim == 2
        assert pi_bt.shape == (batch_size, n_types)
        
        output_dict = dict()
        output_dict['decoder_type'] = decoder_type
        
        if decoder_type == 'posterior_sampled':
            
            assert 'n_samples_per_type' in kwargs
            assert 'max_n_samples_per_type_per_sampling_round' in kwargs
            
            output_dict['weighted_confusion_matrix_btt'] = self.estimate_weighted_confusion_matrix_soft_mc(
                codebook_btls=codebook_btls,
                pi_bt=pi_bt,
                n_samples_per_type=kwargs['n_samples_per_type'],
                max_n_samples_per_type_per_sampling_round=kwargs['max_n_samples_per_type_per_sampling_round'])
            
        elif decoder_type == 'map_reject':
            
            assert 'n_samples_per_type' in kwargs
            assert 'max_n_samples_per_type_per_sampling_round' in kwargs
            assert 'delta_q_max' in kwargs
            assert 'n_map_reject_thresholds' in kwargs
            
            log_prob_map_thresholds_q = get_log_prob_map_thresholds_q(
                n_types=n_types,
                delta_q_max=kwargs['delta_q_max'],
                n_map_reject_thresholds=kwargs['n_map_reject_thresholds'],
                device=self.device,
                dtype=self.dtype)

            confusion_matrix_output_dict = self.estimate_weighted_confusion_matrix_map_reject_mc(
                codebook_btls=codebook_btls,
                pi_bt=pi_bt,
                n_samples_per_type=kwargs['n_samples_per_type'],
                max_n_samples_per_type_per_sampling_round=kwargs['max_n_samples_per_type_per_sampling_round'],
                loq_prob_map_thresholds_q=log_prob_map_thresholds_q)
            
            output_dict['log_prob_map_thresholds_q'] = log_prob_map_thresholds_q
            output_dict['weighted_confusion_matrix_bqtu'] = confusion_matrix_output_dict['weighted_confusion_matrix_bqtu']
            output_dict['num_all_bt'] = confusion_matrix_output_dict['num_all_bt']
            output_dict['num_rej_bqt'] = confusion_matrix_output_dict['num_rej_bqt']
            
        else:
            raise ValueError(
                f'Bad input for decoder_type ({decoder_type}); '
                f'allowed values: posterior_sampled, map_reject')
        
        return output_dict


class BinaryChannelSpecification(NamedTuple):
    # p(SOURCE=0, TARGET=1)
    p_01: float
        
    # p(SOURCE=1, TARGET=0)
    p_10: float


class BinaryAsymmetricChannelModel(SingleEntityChannelModel):
    
    def __init__(
            self,
            channel_spec_list: List[BinaryChannelSpecification],
            device: torch.device,
            dtype: torch.dtype):
        super(BinaryAsymmetricChannelModel, self).__init__(n_symbols=2)
        
        assert isinstance(channel_spec_list, List)            
        assert all(isinstance(entry, BinaryChannelSpecification) for entry in channel_spec_list)
        self.channel_spec_list = channel_spec_list
        
        self.device = device
        self.dtype = dtype
                
        # cache
        self._cache = dict()
        
    @abstractmethod
    def to(self, device: torch.device) -> 'BinaryAsymmetricChannelModel':
        return BinaryAsymmetricChannelModel(
            channel_spec_list=self.channel_spec_list,
            device=device,
            dtype=self.dtype)

    @abstractmethod
    def type(self, dtype: torch.dtype) -> 'BinaryAsymmetricChannelModel':
        return BinaryAsymmetricChannelModel(
            channel_spec_list=self.channel_spec_list,
            device=self.device,
            dtype=dtype)

    @cachedmethod(cache='_cache')
    def get_bool_seq_space_xl(self, code_length: int) -> torch.Tensor:
        bool_seq_space_xl_np = np.zeros((2 ** code_length, code_length), dtype=bool)
        for i_code in range(2 ** code_length):
            bool_seq_space_xl_np[i_code, :] = [int(b) for b in f'{i_code:0{code_length}b}']
        return to_torch(bool_seq_space_xl_np, device=self.device, dtype=torch.bool)
    
    def get_hamming_voronoi_dict(
            self,
            binary_codebook_btl: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculates Voronoi regions for a Hamming decoder."""

        assert binary_codebook_btl.ndim == 3
        binary_codebook_btl = binary_codebook_btl.type(torch.bool)
        batch_size, n_types, code_length = binary_codebook_btl.shape
        
        bool_seq_space_xl = self.get_bool_seq_space_xl(code_length)

        hamming_bxt = (bool_seq_space_xl[None, :, None, :] != binary_codebook_btl[:, None, :, :]).sum(-1)
        torch_min_out = torch.min(hamming_bxt, dim=-1)
        voronoi_bx = torch_min_out.indices
        min_hamming_bx = torch_min_out.values
        
        # reject sequences that are equidistant from 2 or more codes
        multiplicity_bx = (hamming_bxt == min_hamming_bx[..., None]).sum(-1)
        voronoi_bx[multiplicity_bx > 1] = n_types
        
        return {
            'hamming_voronoi_bx': voronoi_bx,
            'hamming_multiplicity_bx': multiplicity_bx
        }   

    def get_broadcasted_channel_spec_list(self, code_length: int) -> List[BinaryChannelSpecification]:
        if len(self.channel_spec_list) == code_length:
            return self.channel_spec_list
        else:
            if len(self.channel_spec_list) == 1:
                return [self.channel_spec_list[0]] * code_length
            else:
                raise ValueError(
                    f"Cannot broadcast the input channel spec list of length {len(channel_spec_list)} "
                    f"to {code_length}!")
        
    def get_seq_log_probs(
            self,
            binary_codebook_btl: torch.Tensor) -> torch.Tensor:
        """Calculates logp(x_n | c_m, channel_specs)"""
        
        assert binary_codebook_btl.ndim == 3
        binary_codebook_btl = binary_codebook_btl.type(torch.bool)
        batch_size, n_types, code_length = binary_codebook_btl.shape
        channel_spec_list = self.get_broadcasted_channel_spec_list(code_length)
        
        bool_seq_space_xl = self.get_bool_seq_space_xl(code_length)
        bool_seq_space_size = bool_seq_space_xl.shape[0]
                
        x_bxtl = bool_seq_space_xl[None, :, None, :]
        c_bxtl = binary_codebook_btl[:, None, :, :]

        t_00_bxtl = (c_bxtl == 0) & (x_bxtl == 0)
        t_01_bxtl = (c_bxtl == 0) & (x_bxtl == 1)
        t_10_bxtl = (c_bxtl == 1) & (x_bxtl == 0)
        t_11_bxtl = (c_bxtl == 1) & (x_bxtl == 1)

        log_p_01_l = to_torch(
            [np.log(channel_spec.p_01) for channel_spec in channel_spec_list],
            device=self.device, dtype=self.dtype)
        log_p_10_l = to_torch(
            [np.log(channel_spec.p_10) for channel_spec in channel_spec_list],
            device=self.device, dtype=self.dtype)
        log_p_00_l = to_torch(
            [np.log(1. - channel_spec.p_01) for channel_spec in channel_spec_list],
            device=self.device, dtype=self.dtype)
        log_p_11_l = to_torch(
            [np.log(1. - channel_spec.p_10) for channel_spec in channel_spec_list],
            device=self.device, dtype=self.dtype)
        
        log_probs_bxt = (
            t_00_bxtl * log_p_00_l +
            t_01_bxtl * log_p_01_l +
            t_10_bxtl * log_p_10_l +
            t_11_bxtl * log_p_11_l).sum(-1)

        return log_probs_bxt

    def get_weighted_confusion_matrix_moffitt(
            self,
            binary_codebook_btl: torch.Tensor,
            pi_bt: torch.Tensor) -> torch.Tensor:
        """Estimates the performance of the Moffitt (2016) decoder from the BAC model.
        
        :param binary_codebook_btl:
        :param pi_bt:
        """

        assert binary_codebook_btl.ndim == 3
        binary_codebook_btl = binary_codebook_btl.type(torch.bool)
        batch_size, n_types, code_length = binary_codebook_btl.shape
        assert pi_bt.ndim == 2
        assert pi_bt.shape == (batch_size, n_types)

        # calculate the Moffitt-style Voronoi region of each of sequence (including rejection)
        hamming_voronoi_dict = self.get_hamming_voronoi_dict(
            binary_codebook_btl=binary_codebook_btl)
        hamming_voronoi_bx = hamming_voronoi_dict['hamming_voronoi_bx']
        
        # calculate log likelihoods for all sequences | codes
        log_probs_bxt = self.get_seq_log_probs(
            binary_codebook_btl=binary_codebook_btl)
        probs_bxt = log_probs_bxt.exp()
        
        # calculate the weighted confusion matrix
        hamming_voronoi_bxu = to_one_hot_encoded(hamming_voronoi_bx, n_types + 1).type(self.dtype)
        weighted_confusion_matrix_btu = torch.einsum(
            'bt,bxt,bxu->btu', pi_bt, probs_bxt, hamming_voronoi_bxu)
        
        return weighted_confusion_matrix_btu
    
    def get_weighted_confusion_matrix_soft(
            self,
            binary_codebook_btl: torch.Tensor,
            pi_bt: torch.Tensor) -> torch.Tensor:
        """Estimates the performance of soft stochastic decoder for a given BAC channel and
        a batch of codebooks.
        """
        
        assert binary_codebook_btl.ndim == 3
        binary_codebook_btl = binary_codebook_btl.type(torch.bool)
        batch_size, n_types, code_length = binary_codebook_btl.shape
        assert pi_bt.ndim == 2
        assert pi_bt.shape == (batch_size, n_types)
        
        # calculate posterior probabilities
        log_probs_bxt = self.get_seq_log_probs(
            binary_codebook_btl=binary_codebook_btl)
        bool_seq_space_size = log_probs_bxt.shape[-2]
        probs_bxt = log_probs_bxt.exp()
        log_pi_bt = pi_bt.log()
        log_posterior_bxt = log_probs_bxt + log_pi_bt[:, None, :]
        log_posterior_bxt = log_posterior_bxt - torch.logsumexp(log_posterior_bxt, dim=-1, keepdim=True)
        posterior_bxt = log_posterior_bxt.exp()
        
        weighted_confusion_matrix_btu = torch.einsum(
            'bt,bxt,bxu->btu', pi_bt, probs_bxt, posterior_bxt)
        
        return weighted_confusion_matrix_btu
        
    def get_weighted_confusion_matrix_map_reject(
            self,
            binary_codebook_btl: torch.Tensor,
            pi_bt: torch.Tensor,
            log_prob_map_thresholds_q: Optional[torch.Tensor]) -> torch.Tensor:
        """Estimates the performance of MAP decoder with reject for a given BAC channel and
        a batch of codebooks.
        """

        # asserts
        if log_prob_map_thresholds_q is not None:
            assert log_prob_map_thresholds_q.ndim == 1
            assert torch.all(log_prob_map_thresholds_q <= 0.)
            n_log_prob_map_thresholds_q = log_prob_map_thresholds_q.shape[0]
        else:
            n_log_prob_map_thresholds_q = 0
        
        assert binary_codebook_btl.ndim == 3
        binary_codebook_btl = binary_codebook_btl.type(torch.bool)
        batch_size, n_types, code_length = binary_codebook_btl.shape
        assert pi_bt.ndim == 2
        assert pi_bt.shape == (batch_size, n_types)
        
        # calculate posterior probabilities
        log_probs_bxt = self.get_seq_log_probs(
            binary_codebook_btl=binary_codebook_btl)
        bool_seq_space_size = log_probs_bxt.shape[-2]
        probs_bxt = log_probs_bxt.exp()
        log_pi_bt = pi_bt.log()
        log_posterior_bxt = log_probs_bxt + log_pi_bt[:, None, :]
        log_posterior_bxt = log_posterior_bxt - torch.logsumexp(log_posterior_bxt, dim=-1, keepdim=True)
        max_out = torch.max(log_posterior_bxt, -1)
        log_posterior_concentration_bx = max_out.values
        
        if n_log_prob_map_thresholds_q > 0:
            # calculate MAP decoder Voronoi regions with reject
            voronoi_bqx = max_out.indices[:, None, :].expand(
                [batch_size, n_log_prob_map_thresholds_q, bool_seq_space_size]).contiguous()
            voronoi_bqx[log_posterior_concentration_bx[:, None, :] < log_prob_map_thresholds_q[None, :, None]] = n_types
            voronoi_bqxu = to_one_hot_encoded(voronoi_bqx, n_types + 1).type(self.dtype)
            weighted_confusion_matrix_bqtu = torch.einsum(
                'bt,bxt,bqxu->bqtu', pi_bt, probs_bxt, voronoi_bqxu)
            return weighted_confusion_matrix_bqtu
        else:
            # calculate MAP decoder Voronoi regions without reject
            voronoi_bx = max_out.indices
            voronoi_bxu = to_one_hot_encoded(voronoi_bx, n_types).type(self.dtype)
            weighted_confusion_matrix_btu = torch.einsum(
                'bt,bxt,bxu->btu', pi_bt, probs_bxt, voronoi_bxu)
            return weighted_confusion_matrix_btu[:, None, :, :]

    def get_weighted_confusion_matrix_moffitt_split(
            self,
            binary_codebook_btl: torch.Tensor,
            pi_bt: torch.Tensor,
            split_size: int) -> torch.Tensor:
        split_output = []
        for _binary_codebook_btl, _pi_bt in split_tensors(
            (binary_codebook_btl, pi_bt), dim=0, split_size=split_size):
            split_output.append(
                self.get_weighted_confusion_matrix_moffitt(
                    binary_codebook_btl=_binary_codebook_btl,
                    pi_bt=_pi_bt))
        return torch.cat(split_output, dim=0)
    
    def get_weighted_confusion_matrix_soft_split(
            self,
            binary_codebook_btl: torch.Tensor,
            pi_bt: torch.Tensor,
            split_size: int) -> torch.Tensor:
        split_output = []
        for _binary_codebook_btl, _pi_bt in split_tensors(
            (binary_codebook_btl, pi_bt), dim=0, split_size=split_size):
            split_output.append(
                self.get_weighted_confusion_matrix_soft(
                    binary_codebook_btl=_binary_codebook_btl,
                    pi_bt=_pi_bt))
        return torch.cat(split_output, dim=0)        
        
    def get_weighted_confusion_matrix_map_reject_split(
            self,
            binary_codebook_btl: torch.Tensor,
            pi_bt: torch.Tensor,
            log_prob_map_thresholds_q: Optional[torch.Tensor],
            split_size: int) -> torch.Tensor:
        split_output = []
        for _binary_codebook_btl, _pi_bt in split_tensors(
            (binary_codebook_btl, pi_bt), dim=0, split_size=split_size):
            split_output.append(
                self.get_weighted_confusion_matrix_map_reject(
                    binary_codebook_btl=_binary_codebook_btl,
                    pi_bt=_pi_bt,
                    log_prob_map_thresholds_q=log_prob_map_thresholds_q))
        return torch.cat(split_output, dim=0)

    def get_weighted_confusion_matrix(
            self,
            codebook_btls: torch.Tensor,
            pi_bt: torch.Tensor,
            decoder_type: str,
            **kwargs) -> Dict[str, torch.Tensor]:
        
        # basic asserts
        assert isinstance(codebook_btls, torch.Tensor)
        assert codebook_btls.ndim == 4
        batch_size, n_types, code_length, n_symbols = codebook_btls.shape
        assert n_symbols == 2
        assert isinstance(pi_bt, torch.Tensor)
        assert pi_bt.ndim == 2
        assert pi_bt.shape == (batch_size, n_types)
        
        output_dict = dict()
        output_dict['decoder_type'] = decoder_type
        
        if decoder_type == 'posterior_sampled':
            
            assert 'split_size' in kwargs
            
            output_dict['weighted_confusion_matrix_btt'] = self.get_weighted_confusion_matrix_soft_split(
                binary_codebook_btl=codebook_btls[..., 1],
                pi_bt=pi_bt,
                split_size=kwargs['split_size'])
            
        elif decoder_type == 'moffitt':
            
            assert 'split_size' in kwargs
            
            output_dict['weighted_confusion_matrix_btu'] = self.get_weighted_confusion_matrix_moffitt_split(
                binary_codebook_btl=codebook_btls[..., 1],
                pi_bt=pi_bt,
                split_size=kwargs['split_size'])

        elif decoder_type == 'map_reject':
            
            assert 'split_size' in kwargs
            assert 'delta_q_max' in kwargs
            assert 'n_map_reject_thresholds' in kwargs
            
            log_prob_map_thresholds_q = get_log_prob_map_thresholds_q(
                n_types=n_types,
                delta_q_max=kwargs['delta_q_max'],
                n_map_reject_thresholds=kwargs['n_map_reject_thresholds'],
                device=self.device,
                dtype=self.dtype)

            output_dict['weighted_confusion_matrix_bqtu'] = self.get_weighted_confusion_matrix_map_reject_split(
                binary_codebook_btl=codebook_btls[..., 1],
                pi_bt=pi_bt,
                log_prob_map_thresholds_q=log_prob_map_thresholds_q,
                split_size=kwargs['split_size'])
            
            output_dict['log_prob_map_thresholds_q'] = log_prob_map_thresholds_q
            output_dict['num_all_bt'] = None
            output_dict['num_rej_bqt'] = None

        else:
            
            raise ValueError(
                f'Bad input for decoder_type ({decoder_type}); '
                f'allowed values: moffitt, posterior_sampled, map_reject')
        
        return output_dict
