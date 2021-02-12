import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, Optional, Set

from mighty_codes.torch_utils import to_np
from mighty_codes.ptpsa import PyTorchBatchedSimulatedAnnealing

# matplotlib plotting configuration
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



def plot_metric_vs_reject(
        metrics_dict: Dict[str, torch.Tensor],
        batch_index: int,
        x_key: str = 'clamped_rej',
        y_key: str = 'clamped_f_1',
        x_label: Optional[str] = 'Rejection Rate',
        y_label: Optional[str] = '$F_1$ Score',
        **plot_kwargs):
    """Plot a metric vs. another for a given metrics dict of a rejection decoder."""

    x_key_full = x_key + '_bqt'
    y_key_full = y_key + '_bqt'
    assert x_key_full in metrics_dict
    assert y_key_full in metrics_dict

    fig = plt.figure()
    ax = plt.gca()
    
    red = np.asarray([1., 0., 0.])
    blue = np.asarray([0., 0., 1.])
    n_types = metrics_dict[x_key_full].shape[-1]
    for i_t in range(n_types):
        x_q = to_np(metrics_dict[x_key_full][batch_index, :, i_t])
        y_q = to_np(metrics_dict[y_key_full][batch_index, :, i_t])
        color = red * (1 - i_t / (n_types - 1)) + blue * i_t / (n_types - 1)
        ax.plot(to_np(x_q), to_np(y_q), color=color, **plot_kwargs)
    ax.set_xlabel(x_key if x_label is None else x_label)
    ax.set_ylabel(y_key if y_label is None else y_label)
    fig.tight_layout()
    
    return fig, ax


def plot_sa_trajectory(
        ptpsa: PyTorchBatchedSimulatedAnnealing,
        output_path: Optional[str] = None,
        output_prefix: Optional[str] = None,
        show: bool = True,
        figsize=(16, 8)):
    i_iter_list = [t.i_iter for t in ptpsa.iteration_summary_list]
    performed_mcmc_list = [t.performed_mcmc for t in ptpsa.iteration_summary_list]
    performed_cooling_list = [t.performed_cooling for t in ptpsa.iteration_summary_list]
    performed_reheating_list = [t.performed_reheating for t in ptpsa.iteration_summary_list]
    performed_resampling_global_list = [t.performed_resampling_global for t in ptpsa.iteration_summary_list]
    performed_resampling_local_list = [t.performed_resampling_local for t in ptpsa.iteration_summary_list]

    def add_events_to_scatter_plot(i_iter_list, event_list, ax, value, **kwargs):
        event_iters = [i_iter_list[idx] for idx in range(len(i_iter_list)) if event_list[idx]]
        ax.scatter(event_iters, [value] * len(event_iters), **kwargs)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2:4, :])
    ax4 = fig.add_subplot(gs[4, :])
    ax5 = fig.add_subplot(gs[5, :])

    add_events_to_scatter_plot(
        i_iter_list, performed_cooling_list,
        ax1,
        1.,
        marker='.', s=20, color='blue', label='cooling')
    add_events_to_scatter_plot(
        i_iter_list, performed_resampling_local_list,
        ax1,
        2.,
        marker='.', s=20, color='green', label='resampling (local)')
    add_events_to_scatter_plot(
        i_iter_list, performed_reheating_list,
        ax1,
        3.,
        marker='x', s=20, color='red', label='reheating')

    ax1.legend(loc='upper right', ncol=3)
    ax1.set_xticks([])
    ax1.set_yticks([])

    i_iter_list = [t.i_iter for t in ptpsa.state_summary_list]
    temperature_list = [1. / t.beta for t in ptpsa.state_summary_list]
    local_energy_min_list = [t.local_energy_min for t in ptpsa.state_summary_list]
    local_energy_max_list = [t.local_energy_max for t in ptpsa.state_summary_list]
    local_energy_mean_list = [t.local_energy_mean for t in ptpsa.state_summary_list]
    global_energy_min_list = [t.global_energy_min for t in ptpsa.state_summary_list]
    mcmc_acc_rate_emp_list = [t.mcmc_acc_rate_emp for t in ptpsa.state_summary_list]
    mcmc_acc_rate_pred_list = [t.mcmc_acc_rate_pred for t in ptpsa.state_summary_list]
    mcmc_acc_rate_lambda_list = [t.mcmc_acc_rate_lambda for t in ptpsa.state_summary_list]

    ax2.plot(i_iter_list, temperature_list, color='red', label='temperature')
    ax2.set_ylabel('temp')
    ax2.set_xticks([])

    ax3.plot(i_iter_list, local_energy_min_list, color='gray', label='min energy (local)')
    ax3.plot(i_iter_list, local_energy_max_list, color='gray', label='max energy (local)')
    ax3.plot(i_iter_list, local_energy_mean_list, color='black', label='mean energy (local)')
    ax3.plot(i_iter_list, global_energy_min_list, color='blue', label='min energy (global)')
    ax3.set_ylabel('energy')
    ax3.set_xticks([])
    ax3.legend(loc='upper right', ncol=4)

    ax4.plot(i_iter_list, mcmc_acc_rate_emp_list, color='green', label='MCMC accept rate (emp)')
    ax4.plot(i_iter_list, mcmc_acc_rate_pred_list, '--', color='green', label='MCMC accept rate (pred)')
    ax4.plot(i_iter_list, [ptpsa.mcmc_eff_acc_rate_lo] * len(i_iter_list), color='gray', lw=0.5)
    ax4.plot(i_iter_list, [ptpsa.mcmc_eff_acc_rate_hi] * len(i_iter_list), color='gray', lw=0.5)
    ax4.set_xlabel('iteration')
    ax4.legend(loc='upper right', ncol=2)

    ax5.plot(i_iter_list, mcmc_acc_rate_lambda_list, color='red', label='MCMC accept rate lambda')
    ax5.set_xlabel('iteration')
    ax5.legend(loc='upper right', ncol=1)
    
    # save
    fig.tight_layout()
    if (output_path is not None) and (output_prefix is not None):
        plt.savefig(os.path.join(output_path, output_prefix + '.pdf'))
    if not show:
        plt.close(fig)

    
def plot_sa_codebook(
        ptpsa: PyTorchBatchedSimulatedAnnealing,
        codebook_bx: torch.Tensor,
        energy_b: torch.Tensor,
        idx: int,
        batch_name: str,
        output_path: Optional[str] = None,
        output_prefix: Optional[str] = None,
        show: bool = True,
        figsize=(12, 8)):
    
    explicit_codebook_btl = ptpsa.batched_state_manipulator.get_explicit_codebook_btl(codebook_bx)
    metrics_dict = ptpsa.batched_state_energy_calculator.calculate_state_metrics_dict(codebook_bx)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(4, 1)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])
    ax4 = fig.add_subplot(gs[3, :])

    ax1.imshow(
        to_np(ptpsa.batched_state_manipulator.get_sorted_codebook_btl(
            explicit_codebook_btl))[idx, :, :].T,
        cmap=plt.cm.Greys_r)
    ax1.set_title(
        f'Batch name: {batch_name}, Index: {idx}, Energy (100x): {100. * energy_b[idx].item():.6f}',
        fontsize=16)
    ax1.set_ylabel('Codeword')
    ax1.set_xlabel('Gene rank')

    ax2.plot(to_np(metrics_dict['tpr_bt'][idx, :]))
    ax2.set_ylabel('TPR')
    ax2.set_xlabel('Gene rank')

    ax3.plot(to_np(metrics_dict['fdr_bt'][idx, :]))
    ax3.set_ylabel('FDR')
    ax3.set_xlabel('Gene rank')

    ax4.plot(to_np(explicit_codebook_btl.sum(-1))[idx, :])
    ax4.set_ylabel('HW')
    ax4.set_xlabel('Gene rank')

    # save
    if (output_path is not None) and (output_prefix is not None):
        plt.savefig(os.path.join(output_path, output_prefix + '.pdf'))
    if not show:
        plt.close(fig)
    

def plot_sa_resampling_buffer(
        ptpsa: PyTorchBatchedSimulatedAnnealing,
        output_path: Optional[str] = None,
        output_prefix: Optional[str] = None,
        show: bool = True,
        figsize=(12, 8)):
    
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    
    for i, t in enumerate(ptpsa.resampling_buffer):
        ax.hist(to_np(t.energy_b), bins=50, alpha=0.5, label=str(i));
    ax.set_xlabel('Energy')
    ax.set_ylabel('Population')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    # save
    fig.tight_layout()
    if (output_path is not None) and (output_prefix is not None):
        plt.savefig(os.path.join(output_path, output_prefix + '.pdf'))
    if not show:
        plt.close(fig)
