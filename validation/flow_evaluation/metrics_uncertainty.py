import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import math

from validation.flow_evaluation.metrics_flow import Fl_kitti_2015, correct_correspondences


def compute_uncertainty_per_image(uncertainty_est, flow_gt, flow_est, mask_valid, dict_list_uncertainties):
    # uncert shape is #number_of_elements
    for uncertainty_name in uncertainty_est.keys():
        if uncertainty_name == 'inference_parameters' or uncertainty_name == 'log_var_map' or \
                uncertainty_name == 'weight_map' or uncertainty_name == 'warping_mask':
            continue

        if 'p_r' == uncertainty_name:
            # convert confidence map to uncertainty
            uncert = (1.0 / (uncertainty_est['p_r'] + 1e-6))[mask_valid.unsqueeze(1)]
        else:
            uncert = uncertainty_est[uncertainty_name][mask_valid.unsqueeze(1)]
        # compute metrics based on uncertainty
        uncertainty_metric_dict = compute_aucs(flow_gt, flow_est, uncert, intervals=50)

        if uncertainty_name not in dict_list_uncertainties.keys():
            # for first image, create the list for each possible uncertainty type
            dict_list_uncertainties[uncertainty_name] = []
        dict_list_uncertainties[uncertainty_name].append(uncertainty_metric_dict)

    return dict_list_uncertainties


def estimate_average_variance_of_mixture_density(log_var_map):
    if isinstance(log_var_map[0], list):
        # several uncertainties estimation
        log_var = log_var_map[0][-1]
        proba_map = torch.nn.functional.softmax(log_var_map[1][-1], dim=1)
    else:
        log_var = log_var_map[0]
        proba_map = torch.nn.functional.softmax(log_var_map[1], dim=1)

    avg_variance = torch.sum(proba_map * torch.exp(log_var), dim=1, keepdim=True) # shape is b,1,  h, w

    R = 1
    var = torch.exp(log_var)
    I = torch.sum(proba_map * (1 - torch.exp(- math.sqrt(2)*R/torch.sqrt(var)))**2, dim=1, keepdim=True)

    return avg_variance, I


def estimate_probability_of_confidence_interval_of_mixture_density(log_var_map, list_item=-1, R = 1.0):
    if isinstance(log_var_map[0], list):
        # several uncertainties estimation
        log_var = log_var_map[0][list_item]
        proba_map = torch.nn.functional.softmax(log_var_map[1][list_item], dim=1)
    else:
        log_var = log_var_map[0]
        proba_map = torch.nn.functional.softmax(log_var_map[1], dim=1)

    var = torch.exp(log_var)
    I = torch.sum(proba_map * (1 - torch.exp(- math.sqrt(2)*R/torch.sqrt(var)))**2, dim=1, keepdim=True)
    return I


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

    if "Fl" in metrics:
        Fl = Fl_kitti_2015(pred, gt) / pred.shape[0] if pred.shape[0] != 0 else 0
        results.append(Fl)

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
    """
    Computation of sparsification curve, oracle curve and auc metric (area below the difference of the two curves),
    for each metrics (AEPE, PCK ..).
    Args:
        gt: gt flow field, shape #number elements, 2
        pred: predicted flow field, shape #number elements, 2
        uncert: predicted uncertainty measure, shape #number elements
        intervals: number of intervals to compute the sparsification plot

    Returns:
        dictionary with sparsification, oracle and AUC for each metric (here EPE, PCK1 and PCK5).
    """
    uncertainty_metrics = ['EPE', 'PCK1', 'PCK5']
    value_for_no_pixels = {'EPE': 0.0, 'PCK1': 1.0, 'PCK5': 1.0}
    # results dictionaries
    AUSE = {'EPE': 0, 'PCK1': 0, 'PCK5': 0}

    # revert order (high uncertainty first)
    uncert = -uncert  # shape #number_elements

    # list the EPE, as the uncertainty. negative because we want high uncertainty first when taking percentile!
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
    return {'sparse_curve': sparse_curve, 'opt_curve': opt_curve, 'AUSE': AUSE}


def compute_average_of_uncertainty_metrics(list_uncertainty_metrics, intervals=50):
    """The sparsification and AUC were computed per image pair. We here compute the average over all image pairs of
    a dataset.
    Args:
        list_uncertainty_metrics: list where each item corresponds to the output of compute_aucs for an image pair.
        intervals: number of intervals to compute the sparsification plot
    Returns:
        dictionary with sparsification, oracle and AUC for each metric (here EPE, PCK1 and PCK5), averaged over all
        elements of the provided list (should correspond to all image pairs of the dataset).
    """
    # https://github.com/mattpoggi/mono-uncertainty/blob/master/evaluate.py
    quants = [1. / intervals * t for t in range(0, intervals + 1)]
    uncertainty_metrics = ['EPE', 'PCK1', 'PCK5']

    keys = list(list_uncertainty_metrics[0].keys())
    output_dict = {m: 0.0 for m in keys}

    for key in keys:
        # list all values
        output_dict[key] = {m: [dict[key][m] for dict in list_uncertainty_metrics] for m in uncertainty_metrics}
        if 'curve' in key:
            # average the curve values
            output_dict[key] = {m: np.array(output_dict[key][m], np.float64).mean(0).tolist()
                                for m in uncertainty_metrics}
        else:
            output_dict[key] = {m: np.array(output_dict[key][m], np.float64).mean() for m in uncertainty_metrics}
    output_dict['quants'] = quants
    '''
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

    '''
    return output_dict
