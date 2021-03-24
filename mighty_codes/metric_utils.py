import torch
import numpy as np

from scipy.sparse import coo_matrix
from typing import List, Tuple, Dict, Dict, Any, Optional, Union
import functools

from mighty_codes.torch_utils import to_np
from mighty_codes.math_utils import log_prob_complement


def get_confusion_matrix_from_indices(
        source_indices: Union[torch.Tensor, np.ndarray],
        target_indices: Union[torch.Tensor, np.ndarray],
        n_classes: int) -> np.ndarray:
    """Given source and target indices, returns a confusion matrix (unnormalized)."""
    source_indices_np = to_np(source_indices)
    target_indices_np = to_np(target_indices)
    data_np = np.ones_like(source_indices_np)
    confusion_matrix = coo_matrix(
        (data_np, (source_indices_np, target_indices_np)),
        shape=(n_classes, n_classes)).todense()
    return np.asarray(confusion_matrix)


def get_metrics_from_confusion_matrix_batched(
        weighted_confusion_matrix_btt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Calculates standard metrics from a batch of confusion matrices.
    
    .. note:
      The "weighted" confusion matrix is the same as the joint probability of
      the source (first dimension) and the target (second dimension).

    :param weighted_confusion_matrix_btt: a tensor of soft (batch_size, n_types, n_types)
    """
    
    # calculate pseudo-counts of different classes
    n_types = weighted_confusion_matrix_btt.shape[-1]
    device = weighted_confusion_matrix_btt.device
    
    # a small number to stabilize calculations
    eps = torch.finfo(weighted_confusion_matrix_btt.dtype).eps
    
    # auxiliary stats
    diag = torch.arange(n_types, device=device)
    tp_bt = weighted_confusion_matrix_btt[:, diag, diag]
    fp_bt = torch.sum(weighted_confusion_matrix_btt, dim=-2) - tp_bt
    fn_bt = torch.sum(weighted_confusion_matrix_btt, dim=-1) - tp_bt
    tn_bt = torch.sum(weighted_confusion_matrix_btt, dim=(-1, -2))[:, None] - tp_bt - fp_bt - fn_bt
    
    # calculate metrics
    tpr_bt = tp_bt / (tp_bt + fn_bt + eps)
    fdr_bt = fp_bt / (tp_bt + fp_bt + eps)
    fpr_bt = fp_bt / (fp_bt + tn_bt + eps)
    acc_bt = (tp_bt + tn_bt) / (tp_bt + tn_bt + fp_bt + fn_bt + eps)
    ppv_bt = 1. - fdr_bt
    f_1_bt = 2 * ppv_bt * tpr_bt / (ppv_bt + tpr_bt + eps)
    
    return {
        'tpr_bt': tpr_bt,
        'fdr_bt': fdr_bt,
        'fpr_bt': fpr_bt,
        'acc_bt': acc_bt,
        'f_1_bt': f_1_bt
    }


def get_metrics_from_confusion_matrix_batched_reject(
        weighted_confusion_matrix_bqtu: torch.Tensor,
        num_rej_bqt: Optional[torch.Tensor],
        num_all_bt: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Calculates standard metrics from a batch of confusion matrices at
    different rejection thresholds.
    
    .. note:
      The "weighted" confusion matrix is the same as the joint probability of
      the source (first dimension) and the target (second dimension).
      
    .. note:
      The very last target (index=n_types) is reserve for rejection.

    :param weighted_confusion_matrix_bqtu: a tensor with shape (batch_size,
      n_rejection_thresholds, n_types, n_types + 1)
    :param num_rej_bqt: (optional) a tensor with shape (batch_size, n_rejection_thresholds,
      n_types) containing the total number of rejected sampled points
    :param num_all_bt: (optional) a tensor with shape (batch_size, n_rejection_thresholds,
      n_types) containing the total number of sampled points
    
    """
    
    # a small number to stabilize calculations
    eps = torch.finfo(weighted_confusion_matrix_bqtu.dtype).eps
    
    # assert shapes
    assert weighted_confusion_matrix_bqtu.ndim >= 3
    assert not ((num_rej_bqt is None) ^ (num_all_bt is None))
    n_types, n_types_p1 = weighted_confusion_matrix_bqtu.shape[-2:]
    assert n_types_p1 == n_types + 1
    
    # true positives
    diag_selector = torch.arange(n_types, device=weighted_confusion_matrix_bqtu.device)
    tp_bqt = weighted_confusion_matrix_bqtu[..., diag_selector, diag_selector]
    
    # predicted positives
    pp_bqt =  weighted_confusion_matrix_bqtu.sum(-2)[..., :n_types]
        
    # condition positives (not rejected)
    cp_bqt = weighted_confusion_matrix_bqtu[..., :n_types].sum(-1)
    
    # all positives (rejected and not rejected)
    ap_bqt = weighted_confusion_matrix_bqtu.sum(-1)

    # false positives
    fp_bqt = pp_bqt - tp_bqt

    # false negatives (in not rejected)
    fn_bqt = cp_bqt - tp_bqt
        
    # true negative (in not rejected)
    tn_bqt = cp_bqt.sum(-1).unsqueeze(-1) - tp_bqt - fp_bqt - fn_bqt
    
    # rejected
    r_bqt = weighted_confusion_matrix_bqtu[..., -1]
    
    # conditional positives and conditional negatives
    pn_bqt = tp_bqt + tn_bqt + fp_bqt + fn_bqt

    # calculate metrics
    ones_bqt = torch.ones_like(tp_bqt)
    zeros_bqt = torch.zeros_like(tp_bqt)
    
    if num_rej_bqt is not None:
        some_not_rejected_bqt = num_rej_bqt < num_all_bt[:, None, :]
        nnz_ap_bqt = (ap_bqt > eps) & some_not_rejected_bqt
        nnz_cp_bqt = (cp_bqt > eps) & some_not_rejected_bqt
        nnz_pp_bqt = (pp_bqt > eps) & some_not_rejected_bqt
        nnz_pn_bqt = (pn_bqt > eps) & some_not_rejected_bqt
    else:
        nnz_ap_bqt = ap_bqt > eps
        nnz_cp_bqt = cp_bqt > eps
        nnz_pp_bqt = pp_bqt > eps
        nnz_pn_bqt = pn_bqt > eps
        
    rej_bqt = torch.where(nnz_ap_bqt, r_bqt / ap_bqt, ones_bqt)
    tpr_bqt = torch.where(nnz_cp_bqt, tp_bqt / cp_bqt, ones_bqt)
    fdr_bqt = torch.where(nnz_pp_bqt, fp_bqt / pp_bqt, zeros_bqt)
    fpr_bqt = torch.where(nnz_cp_bqt, fp_bqt / cp_bqt, zeros_bqt)
    acc_bqt = torch.where(nnz_pn_bqt, (tp_bqt + tn_bqt + eps) / pn_bqt, ones_bqt)
    ppv_bqt = 1. - fdr_bqt
    f_1_bqt = 2 * ppv_bqt * tpr_bqt / (ppv_bqt + tpr_bqt)
    
    return {
        'tpr_bqt': tpr_bqt,
        'fdr_bqt': fdr_bqt,
        'fpr_bqt': fpr_bqt,
        'acc_bqt': acc_bqt,
        'f_1_bqt': f_1_bqt,
        'rej_bqt': rej_bqt
    }

@functools.lru_cache(maxsize=None)
def get_log_prob_map_thresholds_q(
        n_types: int,
        delta_q_max: float,
        n_map_reject_thresholds: int,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
    """Generates log-spaced rejection thresholds to be used for a rejection decoder.
    
    :param n_types: number fo encoded types
    :param delta_q_max: posterior probability concentration complement for the highest
      rejection rate to be considered
    :param n_map_reject_thresholds: number of rejection thresholds
    :param device: torch device
    :param dtype: torch dtype
    """
    
    # generate rejection thresholds
    return log_prob_complement(torch.linspace(
        np.log1p(- 1. / n_types),
        np.log(delta_q_max),
        steps=n_map_reject_thresholds,
        device=device,
        dtype=dtype))


def estimate_codebook_f1_reject_auc(
        weighted_confusion_matrix_bqtu: torch.Tensor,
        num_all_bt: Optional[torch.Tensor],
        num_rej_bqt: Optional[torch.Tensor],
        max_rej_ratio: float,
        interpolation_method: str,
        device: torch.device,
        dtype: torch.device) -> Dict[str, torch.Tensor]:
    """Calculate the area under the curve (AUC) of F_1 score vs. rejection rate curve.
    
    .. note::
      The AUC is normalized to the maximum possible AUC for a given `max_rej_ratio`
      
    .. note::
      The F_1-reject AUC is calculated independently for each type 
    
    .. note::
      Output keys with prefix `clamped_` correspond to outputs clamped at and above `max_rej_ratio`.
      The clamping values are estimated by linear interpolation.

    :param weighted_confusion_matrix_bqtu: weighted confusion matrix with shape (batch_size,
      n_log_prob_map_thresholds, n_types, n_types + 1)
    :param num_all_bt: (optional) number of MC samples with shape (batch_size, n_types)
      (for sampling-based confusion matrix estimators)
    :param num_rej_bqt: (optional) number of rejected MC samples with shape (batch_size,
      n_log_prob_map_thresholds, n_types) (for sampling-based confusion matrix estimators)      
    :param max_rej_ratio: the largest (per-type) rejection rate included in AUC calculation
    :param interpolation_method: lo, linear, hi
    :param device: torch device
    :param dtype: torch dtype
    """
    
    # asserts
    assert weighted_confusion_matrix_bqtu.ndim == 4
    batch_size, n_log_prob_map_thresholds, n_types, n_types_p_1 = weighted_confusion_matrix_bqtu.shape
    assert n_types_p_1 == n_types + 1
    assert not ((num_rej_bqt is None) ^ (num_all_bt is None))
    if num_rej_bqt is not None:
        assert num_rej_bqt.shape == (batch_size, n_log_prob_map_thresholds, n_types)
        assert num_all_bt.shape == (batch_size, n_log_prob_map_thresholds, n_types)
    assert interpolation_method in {'lo', 'hi', 'linear'}

    # a small number to stabilize calculations
    eps = torch.finfo(dtype).eps
    
    # calculate metrics
    metrics_dict = get_metrics_from_confusion_matrix_batched_reject(
        weighted_confusion_matrix_bqtu=weighted_confusion_matrix_bqtu,
        num_all_bt=num_all_bt,
        num_rej_bqt=num_rej_bqt)
    
    # calculate the AUC of F1-reject curve
    f_1_bqt = metrics_dict['f_1_bqt'].clone()
    rej_bqt = metrics_dict['rej_bqt'].clone()
    
    # for each batch and type index, identify the largest q index such that
    # rej_bqt <= max_rej_ratio
    indices_bqt = torch.arange(
        0, n_log_prob_map_thresholds,
        device=device, dtype=torch.int16)[None, :, None].expand(rej_bqt.shape).clone().contiguous()
    indices_bqt[rej_bqt >= max_rej_ratio] = -1    
    lo_bt = indices_bqt.argmax(-2)
    hi_bt = torch.min(lo_bt + 1, n_log_prob_map_thresholds - 1 + torch.torch.zeros_like(lo_bt))

    # calculate f_1 just below and just above rej_ration and interpolate
    rej_lo_bt = torch.gather(rej_bqt, -2, lo_bt[:, None, :])[:, 0, :]
    rej_hi_bt = torch.gather(rej_bqt, -2, hi_bt[:, None, :])[:, 0, :]
    f_1_lo_bt = torch.gather(f_1_bqt, -2, lo_bt[:, None, :])[:, 0, :]
    f_1_hi_bt = torch.gather(f_1_bqt, -2, hi_bt[:, None, :])[:, 0, :]
    if interpolation_method == 'linear':
        f_1_slope_bt = (f_1_hi_bt - f_1_lo_bt) / (rej_hi_bt - rej_lo_bt + eps)
        f_1_interp_bt = f_1_lo_bt + f_1_slope_bt * (max_rej_ratio - rej_lo_bt)
    elif interpolation_method == 'lo':
        f_1_interp_bt = f_1_lo_bt
    elif interpolation_method == 'hi':
        f_1_interp_bt = f_1_hi_bt
    
    # clmap f_1_bqt to f_1_interp_bt for q > hi_bt
    f_1_bqt[rej_bqt >= max_rej_ratio] = f_1_interp_bt[:, None, :].expand(
        rej_bqt.shape)[rej_bqt >= max_rej_ratio]

    # clamp rej_bqt to max_rej_ratio
    rej_bqt[rej_bqt >= max_rej_ratio] = max_rej_ratio
    
    # normalized f_1-reject AUC for each type
    measured_auc_f_1_rej_bt = torch.trapz(f_1_bqt, rej_bqt, dim=-2)
    residual_auc_f_1_rej_bt = f_1_bqt[:, -1, :] * (max_rej_ratio - rej_bqt[:, -1, :])
    normalized_auc_f_1_rej_bt = (measured_auc_f_1_rej_bt + residual_auc_f_1_rej_bt) / max_rej_ratio
    
    # add to the dictionary
    metrics_dict['normalized_auc_f_1_rej_bt'] = normalized_auc_f_1_rej_bt
    metrics_dict['clamped_rej_bqt'] = rej_bqt
    metrics_dict['clamped_f_1_bqt'] = f_1_bqt
    
    return metrics_dict


def get_metrics_dict_from_decoder_output_dict(
        decoder_output_dict: Dict[str, Union[torch.Tensor, str]],
        metrics_dict_type: str,
        **metrics_kwargs) -> Dict[str, torch.Tensor]:
    
    assert 'decoder_type' in decoder_output_dict
    decoder_type = decoder_output_dict['decoder_type']
    assert decoder_type in {'posterior_sampled', 'map_reject'}
    assert metrics_dict_type in {'basic', 'auc'}
    
    if metrics_dict_type == 'basic':
        
        if decoder_type == 'posterior_sampled':
            
            assert 'weighted_confusion_matrix_btt' in decoder_output_dict
            
            return get_metrics_from_confusion_matrix_batched(
                weighted_confusion_matrix_btt=decoder_output_dict['weighted_confusion_matrix_btt'])
            
        elif decoder_type == 'map_reject':
            
            assert 'weighted_confusion_matrix_bqtu' in decoder_output_dict
            assert 'num_rej_bqt' in decoder_output_dict
            assert 'num_all_bt' in decoder_output_dict
            
            return get_metrics_from_confusion_matrix_batched_reject(
                weighted_confusion_matrix_bqtu=decoder_output_dict['weighted_confusion_matrix_bqtu'],
                num_rej_bqt=decoder_output_dict['num_rej_bqt'],
                num_all_bt=decoder_output_dict['num_all_bt'])
    
    if metrics_dict_type == 'auc':
        
        assert decoder_type == 'map_reject'
        assert 'weighted_confusion_matrix_bqtu' in decoder_output_dict
        assert 'num_rej_bqt' in decoder_output_dict
        assert 'num_all_bt' in decoder_output_dict
        assert 'max_rej_ratio' in metrics_kwargs
        assert 'interpolation_method' in metrics_kwargs

        return estimate_codebook_f1_reject_auc(
            weighted_confusion_matrix_bqtu=decoder_output_dict['weighted_confusion_matrix_bqtu'],
            num_rej_bqt=decoder_output_dict['num_rej_bqt'],
            num_all_bt=decoder_output_dict['num_all_bt'],
            max_rej_ratio=metrics_kwargs['max_rej_ratio'],
            interpolation_method=metrics_kwargs['interpolation_method'],
            device=decoder_output_dict['weighted_confusion_matrix_bqtu'].device,
            dtype=decoder_output_dict['weighted_confusion_matrix_bqtu'].dtype)
    
    raise RuntimeError('Should not reach here!')


def parse_quantile_based_metric_type_str(
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
    

def get_top_quantile(quantile: float, metric_bt: torch.Tensor) -> torch.Tensor:
    n_types = metric_bt.size(1)
    n_keep = math.ceil(n_types * quantile)
    sorted_metric_bt = torch.sort(metric_bt, dim=-1).values
    return sorted_metric_bt[:, -n_keep:]


def get_optimality_from_metrics_dict(
        metrics_dict: Dict[str, torch.Tensor],
        optimality_type: str) -> torch.Tensor:
    
    mean_reduce = lambda metric_bt: metric_bt.mean(-1)
    complement = lambda metric_b: (1. - metric_b)
    optional_fdr_quantile = parse_quantile_based_metric_type_str('fdr', optimality_type)
    optional_tpr_quantile = parse_quantile_based_metric_type_str('tpr', optimality_type)

    if optimality_type == 'f1_reject_auc':
        return mean_reduce(
            metrics_dict['normalized_auc_f_1_rej_bt'])

    elif optimality_type == 'fdr':
        return mean_reduce(
            complement(
                metrics_dict['fdr_bt']))

    elif optimality_type == 'tpr':
        return mean_reduce(
            metrics_dict['tpr_bt'])

    elif optional_fdr_quantile is not None:
        return mean_reduce(
            complement(
                get_top_quantile(
                    quantile=optional_fdr_quantile,
                    metric_bt=metrics_dict['fdr_bt'])))

    elif optional_tpr_quantile is not None:
        return mean_reduce(
            get_top_quantile(
                quantile=optional_fdr_quantile,
                metric_bt=metrics_dict['tpr_bt']))
    else:
        
        raise ValueError(
            'Unknown metric type -- allowed values: '
            'f1_reject_auc, fdr, tpr, fdr[<quantile>], tpr[<quantile>]')