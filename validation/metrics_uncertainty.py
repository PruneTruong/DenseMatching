import torch
import numpy as np
from matplotlib import pyplot as plt
from validation.metrics_flow import correct_correspondences
import math


def save_sparsification_plot(output, save_path):
    EPE_uncer = output["sparse_curve"]["EPE"]
    EPE_perfect = output["opt_curve"]["EPE"]
    x = output["quants"]

    fig, axis = plt.subplots(1, 1, figsize=(25, 25))
    axis.plot(x, np.float32(EPE_uncer) / EPE_uncer[0], marker='o', markersize=13, linewidth=5, label='estimated uncertainty')
    axis.plot(x, np.float32(EPE_perfect) / EPE_uncer[0], marker='o', markersize=13, linewidth=5, label='optimal')
    axis.set_ylabel('EPE normalized', fontsize='xx-large')
    axis.set_xlabel('Removing x fraction of pixels', fontsize='xx-large')
    axis.legend(bbox_to_anchor=(1.05, 1), loc=4, borderaxespad=0., fontsize='xx-large')
    fig.savefig('{}.png'.format(save_path),
                bbox_inches='tight')
    plt.close(fig)


def compute_eigen_errors_v2(gt, pred, metrics=['EPE', 'PCK1', 'PCK5'], mask=None, reduce_mean=True):
    """Revised compute_eigen_errors function used for uncertainty metrics, with optional reduce_mean argument and (1-a1) computation
    """
    results = []

    # in shape (#number_elements, 2)
    # mask shape #number_of_elements
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    if "EPE" in metrics:
        epe = torch.norm(gt - pred, p=2, dim=1)
        if reduce_mean:
            epe = epe.mean().item()
        results.append(epe)

    if "PCK1" in metrics:
        px_1 = correct_correspondences(pred, gt, alpha=1.0, img_size=1.0)
        pck1 = px_1 / (pred.shape[0]) if pred.shape[0] != 0 else 0
        results.append(pck1)

    if "PCK5" in metrics:
        px_5 = correct_correspondences(pred, gt, alpha=5.0, img_size=1.0)
        pck5 = px_5 / (pred.shape[0]) if pred.shape[0] != 0 else 0
        results.append(pck5)

    return results


def compute_aucs(gt, pred, uncert, intervals=50):
    """Computation of auc metrics
    gt and pred shape #number elements, 2
    uncertainty shape #number of elements
    """
    uncertainty_metrics = ['EPE', 'PCK1', 'PCK5']
    value_for_no_pixels = {'EPE': 0.0, 'PCK1': 1.0, 'PCK5': 1.0}
    # results dictionaries
    AUSE = {'EPE': 0, 'PCK1': 0, 'PCK5': 0}

    # revert order (high uncertainty first)
    uncert = -uncert # shape #number_elements

    # list the EPE, as the uncertainty. negative because we want high uncertainty first !
    true_uncert = - torch.norm(gt - pred, p=2, dim=1)

    # prepare subsets for sampling and for area computation
    quants = [100. / intervals * t for t in range(0, intervals)]
    plotx = [1. / intervals * t for t in range(0, intervals + 1)]

    # get percentiles for sampling and corresponding subsets
    thresholds = [np.percentile(uncert.cpu().numpy(), q) for q in quants]
    subs = [(uncert.ge(t)) for t in thresholds]

    # compute sparsification curves for each metric (add 0 for final sampling)
    # calculates the metrics for each interval
    sparse_curve = {
        m: [compute_eigen_errors_v2(gt, pred, metrics=[m], mask=sub, reduce_mean=True)[0] for sub in subs] +
           [value_for_no_pixels[m]] for m in uncertainty_metrics}

    # human-readable call
    '''
    sparse_curve =  {"rmse":[compute_eigen_errors_v2(gt,pred,metrics=["rmse"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0], 
                     "a1":[compute_eigen_errors_v2(gt,pred,metrics=["a1"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0],
                     "abs_rel":[compute_eigen_errors_v2(gt,pred,metrics=["abs_rel"],mask=sub,reduce_mean=True)[0] for sub in subs]+[0]}
    '''

    # get percentiles for optimal sampling and corresponding subsets (based on real EPE)
    opt_thresholds = [np.percentile(true_uncert.cpu().numpy(), q) for q in quants]
    opt_subs = [(true_uncert.ge(o)) for o in opt_thresholds]

    # compute sparsification curves for optimal sampling (add 0 for final sampling)
    opt_curve = {m: [compute_eigen_errors_v2(gt, pred, metrics=[m], mask=opt_sub, reduce_mean=True)[0] for opt_sub in
                     opt_subs] + [value_for_no_pixels[m]] for m in uncertainty_metrics}

    # compute error and gain metrics
    for m in uncertainty_metrics:
        max = np.array(opt_curve[m]).max() + 1e-6
        # normalize both to 0-1 first
        # error: subtract from method sparsification (first term) the oracle sparsification (second term)
        AUSE[m] = np.abs(np.trapz(np.array(sparse_curve[m])/max, x=plotx) -
                         np.trapz(np.array(opt_curve[m])/max, x=plotx))

        opt_curve[m] = np.array(opt_curve[m]) / max
        sparse_curve[m] = np.array(sparse_curve[m]) / max  # normalizes each curve to 0 and 1
    # returns a dictionary with AUSE and AURG for each metric
    return {'sparse_curve': sparse_curve,
            'opt_curve': opt_curve,
            'AUSE': AUSE}


def compute_average_of_uncertainty_metrics(list_uncertainty_metrics, intervals=50):
    # https://github.com/mattpoggi/mono-uncertainty/blob/master/evaluate.py
    quants = [1. / intervals * t for t in range(0, intervals + 1)]
    uncertainty_metrics = ['EPE', 'PCK1', 'PCK5']
    output_dict = {'sparse_curve':0, 'opt_curve': 0, 'AUSE': 0}

    output_dict['sparse_curve'] = {m: [dict['sparse_curve'][m] for dict in list_uncertainty_metrics]
                                   for m in uncertainty_metrics}
    output_dict['opt_curve'] = {m: [dict['opt_curve'][m] for dict in list_uncertainty_metrics]
                                for m in uncertainty_metrics}
    output_dict['AUSE'] = {m: [dict['AUSE'][m] for dict in list_uncertainty_metrics] for m in uncertainty_metrics}

    output_dict['sparse_curve'] = {m: np.array(output_dict['sparse_curve'][m], np.float64).mean(0).tolist()
                                   for m in uncertainty_metrics}
    output_dict['opt_curve'] = {m: np.array(output_dict['opt_curve'][m], np.float64).mean(0).tolist() for m in
                                uncertainty_metrics}
    output_dict['AUSE'] = {m: np.array(output_dict['AUSE'][m], np.float64).mean() for m in uncertainty_metrics}
    output_dict['quants'] = quants
    return output_dict
