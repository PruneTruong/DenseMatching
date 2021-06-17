import numpy as np
import os
import torch.nn as nn
from utils_flow.util_optical_flow import flow_to_image
import torch.nn.functional as F
import torch
from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from matplotlib import pyplot as plt
from datasets.util import draw_matches


def plot_only_images_and_flow(source_image, target_image, remapped_est, rgb_es_flow=None, remapped_gt=None, rgb_gt_flow=None,
                              mask=None, compute_rgb_flow=False):
    if mask is not None:
        fig, axis = plt.subplots(1, 5, figsize=(20, 10))
        axis[4].imshow(mask.squeeze().cpu().numpy())
        axis[4].set_title("mask")
    else:
        fig, axis = plt.subplots(1, 4, figsize=(20, 10))
    axis[0].imshow(source_image)
    axis[0].set_title("source image")
    axis[1].imshow(target_image)
    axis[1].set_title("target image")
    if compute_rgb_flow:
        axis[2].imshow(rgb_es_flow)
        axis[2].set_title("source remapped with network")
        if rgb_gt_flow is not None:
            axis[3].imshow(rgb_gt_flow)
            axis[3].set_title("source remapped with gt")
    else:
        axis[2].imshow(remapped_est)
        axis[2].set_title("source remapped with network")
        if remapped_gt is not None:
            axis[3].imshow(remapped_gt)
            axis[3].set_title("source remapped with gt")
    return fig


def plot_flow_and_uncertainty(path_to_save, name_image, source_image, target_image, gt_flow, estimated_flow,
                              uncertainty_comp_est=None, mask=None, compute_rgb_flow=False):

    if not isinstance(source_image, np.ndarray):
        source_image = source_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        target_image = target_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        if source_image.shape[1] == 3:
            source_image = source_image.transpose(0, 2, 3, 1)
            target_image = target_image.transpose(0, 2, 3, 1)
        source_image = source_image[0].astype(np.uint8)
        target_image = target_image[0].astype(np.uint8)
    h, w, _ = target_image.shape

    if not isinstance(estimated_flow, np.ndarray):
        estimated_flow = estimated_flow.permute(0, 2, 3, 1)[0].cpu().numpy()
    if not isinstance(gt_flow, np.ndarray):
        gt_flow = gt_flow.permute(0, 2, 3, 1)[0].cpu().numpy()
    rgb_gt_flow = flow_to_image(gt_flow)
    rgb_es_flow = flow_to_image(estimated_flow)

    remapped_est = remap_using_flow_fields(source_image, estimated_flow[:,:,0], estimated_flow[:,:,1]).astype(np.uint8)
    remapped_gt = remap_using_flow_fields(source_image, gt_flow[:,:,0], gt_flow[:,:,1]).astype(np.uint8)

    if uncertainty_comp_est is None:
        fig = plot_only_images_and_flow(source_image, target_image, remapped_est, rgb_es_flow,
                                        remapped_gt=remapped_gt, rgb_gt_flow=rgb_gt_flow, mask=mask,
                                        compute_rgb_flow=compute_rgb_flow)
    else:
        log_var = uncertainty_comp_est['log_var_map']
        weight_map = uncertainty_comp_est['weight_map']
        proba_map = torch.nn.functional.softmax(weight_map.detach(), dim=1)
        n = 2
        num_mixture_mode = proba_map.shape[1]  # number of different modes

        fig, axis = plt.subplots(n, num_mixture_mode * 2 + 2, figsize=(25, 10))
        for ind_mode in range(num_mixture_mode):
            proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
            axis[1][ind_mode * 2].set_title("Laplace component {}: Proba map \n, min={:.2f}, max={:.2f}"
                                            .format(ind_mode, round(proba_map_numpy.min(), 2),
                                                    round(proba_map_numpy.max(), 2)))
            log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[1][ind_mode * 2 + 1].imshow(log_var_numpy)
            axis[1][ind_mode * 2 + 1].set_title("Laplace component {}: Log Variance:\n var min={:.2f}, max={:.2f}"
                                                .format(ind_mode, round(np.exp(log_var_numpy).min(), 2),
                                                        round(np.exp(log_var_numpy).max(), 2)))

        avg_variance = uncertainty_comp_est['variance'].squeeze().cpu().numpy().astype(np.float32)
        axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=50.0)
        axis[1][num_mixture_mode * 2].set_title("variance of the mixture,\n min={}, max={}"
                                                .format(round(avg_variance.min()),
                                                        round(avg_variance.max())))

        probability_interval = uncertainty_comp_est['p_r'].squeeze().cpu().numpy().squeeze()
        axis[1][num_mixture_mode * 2 + 1].imshow(probability_interval, vmin=0, vmax=1.0)
        axis[1][num_mixture_mode * 2 + 1].set_title(
            "Proba of confidence value,\n min={}, max={}".format(
                round(probability_interval.min() * 100),
                round(probability_interval.max() * 100)))

        axis[0][0].imshow(source_image)
        axis[0][0].set_title("source image")
        axis[0][1].imshow(target_image)
        axis[0][1].set_title("target image")
        axis[0][2].imshow(remapped_gt)
        axis[0][2].set_title("source remapped with ground-truth")
        axis[0][3].imshow(remapped_est)
        axis[0][3].set_title("source remapped with network")
        alpha = 0.5
        axis[0][4].imshow((remapped_est * alpha + (1.0 - alpha) * target_image).astype(np.uint8))
        axis[0][4].set_title("source remapped with network overlaid with target")
        if mask is not None:
            mask = mask.squeeze().cpu().numpy()
            axis[0][5].imshow(remapped_est * np.tile(np.expand_dims(mask.astype(np.uint8), axis=2), (1, 1, 3)))
            axis[0][5].set_title("source remapped with network")

    # save figure
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_save, name_image + '.jpg'))  # save the figure to file
    plt.close(fig)


def plot_sparse_keypoints(path_to_save, name_image, source_image, target_image, estimated_flow, Xs, Ys, Xt, Yt,
                          uncertainty_comp_est=None, mask=None, compute_rgb_flow=False):

    if not isinstance(source_image, np.ndarray):
        source_image = source_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        target_image = target_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        if source_image.shape[1] == 3:
            source_image = source_image.transpose(0, 2, 3, 1)
            target_image = target_image.transpose(0, 2, 3, 1)
        source_image = source_image[0].astype(np.uint8)
        target_image = target_image[0].astype(np.uint8)
    h, w, _ = target_image.shape

    if not isinstance(estimated_flow, np.ndarray):
        estimated_flow = estimated_flow.permute(0, 2, 3, 1)[0].cpu().numpy()

    rgb_es_flow = flow_to_image(estimated_flow)

    remapped_est = remap_using_flow_fields(source_image, estimated_flow[:, :, 0],
                                           estimated_flow[:, :, 1]).astype(np.uint8)
    image_matches_gt = np.clip(draw_matches(source_image, target_image,
                                            np.concatenate((Xs.reshape(len(Xs), 1), Ys.reshape(len(Xs), 1)), axis=1),
                                            np.concatenate((Xt.reshape(len(Xs), 1), Yt.reshape(len(Xs), 1)), axis=1))
                               .astype(np.uint8), 0, 255)

    if uncertainty_comp_est is None:
        fig = plot_only_images_and_flow(source_image, target_image, remapped_est, rgb_es_flow, mask=mask,
                                        compute_rgb_flow=compute_rgb_flow)
    else:
        log_var = uncertainty_comp_est['log_var_map']
        weight_map = uncertainty_comp_est['weight_map']
        proba_map = torch.nn.functional.softmax(weight_map.detach(), dim=1)
        n = 2
        num_mixture_mode = proba_map.shape[1]  # number of different modes

        fig, axis = plt.subplots(n, num_mixture_mode * 2 + 2, figsize=(25, 10))
        for ind_mode in range(num_mixture_mode):
            proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
            axis[1][ind_mode * 2].set_title("Laplace component {}: Proba map \n, min={:.2f}, max={:.2f}"
                                            .format(ind_mode, round(proba_map_numpy.min(), 2),
                                                    round(proba_map_numpy.max(), 2)))
            log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[1][ind_mode * 2 + 1].imshow(log_var_numpy)
            axis[1][ind_mode * 2 + 1].set_title("Laplace component {}: Log Variance:\n var min={:.2f}, max={:.2f}"
                                                .format(ind_mode, round(np.exp(log_var_numpy).min(), 2),
                                                        round(np.exp(log_var_numpy).max(), 2)))

        avg_variance = uncertainty_comp_est['variance'].squeeze().cpu().numpy().astype(np.float32)
        axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=50.0)
        axis[1][num_mixture_mode * 2].set_title("variance of the mixture,\n min={}, max={}"
                                                .format(round(avg_variance.min()),
                                                        round(avg_variance.max())))

        probability_interval = uncertainty_comp_est['p_r'].squeeze().cpu().numpy().squeeze()
        axis[1][num_mixture_mode * 2 + 1].imshow(probability_interval, vmin=0, vmax=1.0)
        axis[1][num_mixture_mode * 2 + 1].set_title(
            "Proba of confidence value,\n min={}, max={}".format(
                round(probability_interval.min() * 100),
                round(probability_interval.max() * 100)))

        axis[0][0].imshow(source_image)
        axis[0][0].set_title("source image")
        axis[0][1].imshow(target_image)
        axis[0][1].set_title("target image")
        axis[0][2].imshow(image_matches_gt)
        axis[0][2].set_title("Ground-truth matches")
        axis[0][3].imshow(remapped_est)
        axis[0][3].set_title("source remapped with network")
        alpha = 0.5
        axis[0][4].imshow((remapped_est * alpha + (1.0 - alpha) * target_image).astype(np.uint8))
        axis[0][4].set_title("source remapped with network overlaid with target")
        if mask is not None:
            mask = mask.squeeze().cpu().numpy()
            axis[0][5].imshow(remapped_est * np.tile(np.expand_dims(mask.astype(np.uint8), axis=2), (1, 1, 3)))
            axis[0][5].set_title("source remapped with network")

    # save figure
    fig.tight_layout()
    fig.savefig(os.path.join(path_to_save, name_image + '.jpg'))  # save the figure to file
    plt.close(fig)
