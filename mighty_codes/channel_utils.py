import numpy as np
import torch

from mighty_codes.channels import BinaryAsymmetricChannelModel
import mighty_codes.metric_utils as metric_utils


def calculate_bac_f1_reject_auc_metric_dict(
        codebook_btl: torch.Tensor,
        pi_t: torch.Tensor,
        bac_model: BinaryAsymmetricChannelModel,
        delta_q_max: float,
        n_map_reject_thresholds: int,
        max_reject_ratio: float,
        split_size: int,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
    
    assert codebook_btl.ndim == 3
    batch_size, n_types, code_length = codebook_btl.shape
    
    log_prob_map_thresholds_q = metric_utils.get_log_prob_map_thresholds_q(
        n_types=n_types,
        delta_q_max=delta_q_max,
        n_map_reject_thresholds=n_map_reject_thresholds,
        device=device,
        dtype=dtype)
        
    # calculate confusion matrix
    pi_bt = pi_t.expand([batch_size, n_types])
    weighted_confusion_matrix_bqtu = bac_model.get_weighted_confusion_matrix_map_reject_split(
        binary_codebook_btl=codebook_btl,
        pi_bt=pi_bt,
        log_prob_map_thresholds_q=log_prob_map_thresholds_q,
        split_size=split_size)
    
    # calculate optimality metric
    metrics_dict = metric_utils.estimate_codebook_f1_reject_auc(
        weighted_confusion_matrix_bqtu=weighted_confusion_matrix_bqtu,
        num_all_bt=None,
        num_rej_bqt=None,
        max_rej_ratio=max_reject_ratio,
        interpolation_method='lo',
        device=device,
        dtype=dtype)
    
    return metrics_dict


def calculate_bac_standard_metric_dict(
        codebook_btl: torch.Tensor,
        pi_t: torch.Tensor,
        bac_model: BinaryAsymmetricChannelModel,
        split_size: int) -> torch.Tensor:
    
    assert codebook_btl.ndim == 3
    batch_size, n_types, code_length = codebook_btl.shape

    # calculate confusion matrix
    pi_bt = pi_t.expand([batch_size, n_types])
    weighted_confusion_matrix_btt = bac_model.get_weighted_confusion_matrix_soft_split(
        binary_codebook_btl=codebook_btl,
        pi_bt=pi_bt,
        split_size=split_size)
    
    # calculate optimality metric
    metrics_dict = metric_utils.get_metrics_from_confusion_matrix_batched(
        weighted_confusion_matrix_btt=weighted_confusion_matrix_btt)
    
    return metrics_dict

