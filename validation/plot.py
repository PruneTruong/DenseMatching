import numpy as np
import os
import imageio
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from imageio import imwrite

from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.visualization_utils import draw_matches
from validation.flow_evaluation.metrics_uncertainty import estimate_probability_of_confidence_interval_of_mixture_density
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping
from utils_flow.visualization_utils import make_and_save_video, overlay_semantic_mask, replace_area


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


def plot_occlusion(source_image, target_image, gt_flow, estimated_flow, warping_mask=None, occ_mask=None, gt_occ_mask=None, path_to_save=None):
    rgb_gt_flow = flow_to_image(gt_flow)
    rgb_es_flow = flow_to_image(estimated_flow)
    fig, axis = plt.subplots(2, 3, figsize=(25, 25))  # create figure & 1 axis
    axis[0][0].imshow(source_image)
    axis[0][0].set_title('source image')
    axis[0][1].imshow(target_image)
    axis[0][1].set_title('target image')
    axis[0][2].imshow(rgb_gt_flow)
    axis[0][2].set_title('gt flow')
    axis[1][0].imshow(rgb_es_flow)
    axis[1][0].set_title('estimated flow')
    if gt_occ_mask is not None:
        axis[1][1].imshow(gt_occ_mask.squeeze().cpu().numpy(), vmin=0, vmax=1)
        axis[1][1].set_title("GT Occlusion mask")

    if warping_mask is not None:
        warping_mask = (F.sigmoid(warping_mask)).squeeze().cpu().numpy()
        axis[1][1].imshow(warping_mask, vmin=0, vmax=1)
        axis[1][1].set_title("Warping, min = {}, max={}".format(round(warping_mask.min(),2),
                                                              round(warping_mask.max(),2)))
    if occ_mask is not None:
        occlusion_mask = (F.sigmoid(occ_mask)).squeeze().cpu().numpy()
        axis[1][2].imshow(occlusion_mask, vmin=0, vmax=1)
        axis[1][2].set_title("Occlusion mask")
    fig.savefig(path_to_save, bbox_inches='tight')  # save the figure to file
    plt.close(fig)


def plot_flow_and_uncertainty(path_to_save, name_image, source_image, target_image, gt_flow, estimated_flow,
                              uncertainty_comp_est=None, plot_log_var=False,
                              save_dir=None, mask=None, compute_rgb_flow=False):

    if plot_log_var:
        if not os.path.isdir(os.path.join(save_dir, 'individual_variances')):
            os.makedirs(os.path.join(save_dir, 'individual_variances'))

    if not isinstance(source_image, np.ndarray):
        source_image = source_image.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
        target_image = target_image.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
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

        if 'weight_map' in list(uncertainty_comp_est.keys()):
            # means you have multiple components, it is a mixture model
            log_var = uncertainty_comp_est['log_var_map']
            weight_map = uncertainty_comp_est['weight_map']
            proba_map = torch.nn.functional.softmax(weight_map, dim=1)
            n = 2
            num_mixture_mode = proba_map.shape[1]  # number of different modes
            fig, axis = plt.subplots(n, num_mixture_mode * 2 + 2, figsize=(25, 10))
            for ind_mode in range(num_mixture_mode):
                proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                axis[1][ind_mode * 2].set_title("Var Mode {} Proba map \n, min={:.2f}, max={:.2f}".format(ind_mode,
                                                                                                  round(
                                                                                                      proba_map_numpy.min(), 2),
                                                                                                  round(
                                                                                                      proba_map_numpy.max(), 2)))
                log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1][ind_mode * 2 + 1].imshow(log_var_numpy)
                axis[1][ind_mode * 2 + 1].set_title("Var Mode {} Log Variance:\n var min={:.2f}, max={:.2f}"
                                                    .format(ind_mode, round(np.exp(log_var_numpy).min(), 2),
                                                            round(np.exp(log_var_numpy).max(), 2)))

                if plot_log_var:
                    fig_var, ax = plt.subplots()
                    plt.axis('off')
                    ax.imshow(log_var_numpy)
                    fig_var.savefig(os.path.join(save_dir, 'individual_variances',
                                                 "{}_variance_{}.png".format(name_image, ind_mode)),
                                    bbox_inches='tight', pad_inches=0)
                    plt.close(fig_var)

                    fig_proba, ax = plt.subplots()
                    plt.axis('off')
                    ax.imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                    fig_proba.savefig(os.path.join(save_dir, 'individual_variances',
                                                   "{}_proba_map_{}.png".format(name_image, ind_mode)),
                                      bbox_inches='tight', pad_inches=0)
                    plt.close(fig_proba)

            avg_variance = uncertainty_comp_est['variance'].squeeze().cpu().numpy().astype(np.float32)
            axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
            axis[1][num_mixture_mode * 2].set_title("variance of the mixture\n, min={}, max={}"
                                                    .format(round(avg_variance.min()),
                                                            round(avg_variance.max())))

            probability_interval = uncertainty_comp_est['p_r'].cpu().numpy().squeeze().astype(np.float32)
            axis[1][num_mixture_mode * 2 + 1].imshow(probability_interval, vmin=0, vmax=1.0)
            axis[1][num_mixture_mode * 2 + 1].set_title(
                "proba of confidence value within 1\n min={}, max={}".format(
                    round(probability_interval.min() * 100),
                    round(probability_interval.max() * 100)))

            if plot_log_var:
                fig_proba, ax = plt.subplots()
                plt.axis('off')
                ax.imshow(probability_interval, vmin=0.0, vmax=1.0)
                fig_proba.savefig(
                    os.path.join(save_dir, 'individual_variances', "{}_proba_R_1.png".format(name_image)),
                    bbox_inches='tight', pad_inches=0)
                plt.close(fig_proba)

                probability_interval_3 = estimate_probability_of_confidence_interval_of_mixture_density(log_var_map,
                                                                                                        R=3.0).cpu().numpy().squeeze()
                fig_proba, ax = plt.subplots()
                plt.axis('off')
                ax.imshow(probability_interval_3, vmin=0.0, vmax=1.0)
                fig_proba.savefig(
                    os.path.join(save_dir, 'individual_variances', "{}_proba_R_3.png".format(name_image)),
                    bbox_inches='tight', pad_inches=0)
                plt.close(fig_proba)

                probability_interval_5 = estimate_probability_of_confidence_interval_of_mixture_density(log_var_map,
                                                                                                        R=5.0).cpu().numpy().squeeze()
                fig_proba, ax = plt.subplots()
                plt.axis('off')
                ax.imshow(probability_interval_5, vmin=0.0, vmax=1.0)
                fig_proba.savefig(
                    os.path.join(save_dir, 'individual_variances', "{}_proba_R_5.png".format(name_image)),
                    bbox_inches='tight', pad_inches=0)
                plt.close(fig_proba)

            axis[0][0].imshow(source_image)
            axis[0][0].set_title('source image')
            axis[0][1].imshow(target_image)
            axis[0][1].set_title('target image')
            axis[0][2].imshow(rgb_gt_flow)
            axis[0][2].set_title('gt flow')
            axis[0][3].imshow(rgb_es_flow)
            axis[0][3].set_title('estimated flow')
            axis[0][4].imshow(remapped_gt)
            axis[0][4].set_title('source remapped with gt flow')
            axis[0][5].imshow(remapped_est)
            axis[0][5].set_title('source remapped with estimated flow')

        else:
            fig, axis = plt.subplots(2, 4, figsize=(25, 10))  # create figure & 1 axis
            image = uncertainty_comp_est['log_var_map'].squeeze().cpu().numpy()
            axis[1][2].imshow(image)
            axis[1][2].set_title('log variance. \n var max is {}, min {}'.format(round(np.exp(image).max(),2),
                                                                               round(np.exp(image).min(),2)))

            axis[0][0].imshow(source_image)
            axis[0][0].set_title('source image')
            axis[0][1].imshow(target_image)
            axis[0][1].set_title('target image')
            axis[0][2].imshow(rgb_gt_flow)
            axis[0][2].set_title('gt flow')
            axis[0][3].imshow(rgb_es_flow)
            axis[0][3].set_title('estimated flow')
            axis[1][0].imshow(remapped_gt)
            axis[1][0].set_title('source remapped with gt flow')
            axis[1][1].imshow(remapped_est)
            axis[1][1].set_title('source remapped with estimated flow')

    fig.tight_layout()
    fig.savefig(os.path.join(path_to_save, name_image + '.jpg'))  # save the figure to file
    plt.close(fig)


def plot_sparse_keypoints(save_path, name_image, source_image, target_image, flow_est, Xs, Ys, Xt, Yt,
                          warping_mask=None, occlusion_mask=None, uncertainty_comp_est=None,
                          plot_log_var=False, mask_used=None):
    if plot_log_var:
        if not os.path.isdir(os.path.join(save_path, 'individual_variances')):
            os.makedirs(os.path.join(save_path, 'individual_variances'))

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

    flow_target = flow_est.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    remapped_est = remap_using_flow_fields(source_image, flow_target[:, :, 0], flow_target[:, :, 1]).astype(np.uint8)

    image_matches_gt = np.clip(draw_matches(source_image, target_image, np.concatenate((Xs.reshape(len(Xs), 1),
                                                                                        Ys.reshape(len(Xs), 1)), axis=1),
                                            np.concatenate((Xt.reshape(len(Xs), 1), Yt.reshape(len(Xs), 1)),
                                                           axis=1)).astype(np.uint8), 0, 255)
    imageio.imwrite(os.path.join(save_path, 'matches_gt_{}.jpg'.format(name_image)), image_matches_gt)

    if occlusion_mask is None and warping_mask is None and uncertainty_comp_est is None:
        # only output the flow
        if mask_used is not None:
            fig, axis = plt.subplots(1, 5, figsize=(20, 20))
            axis[4].imshow(mask_used.squeeze().cpu().numpy())
            axis[4].set_title("mask")
        else:
            fig, axis = plt.subplots(1, 4, figsize=(20, 20))
        axis[0].imshow(source_image)
        axis[0].set_title("source image")
        axis[1].imshow(target_image)
        axis[1].set_title("target image")
        axis[2].imshow(remapped_est)
        axis[2].set_title("source remapped with network")
        axis[3].imshow(image_matches_gt)
        axis[3].set_title("image matches gt, nbr_matches={}".format(len(Xs)))
        '''
        if not os.path.isdir(os.path.join(save_path, 'individual_images')):
            os.makedirs(os.path.join(save_path, 'individual_images'))
        imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_image.png".format(name_image)),
                        image_2)
        imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_remapped.png".format(name_image)),
                        remapped_est)
        imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_remapped_alpha.png".format(name_image)), remapped_alpha)
        '''
    else:
        if uncertainty_comp_est is not None and 'weight_map' in list(uncertainty_comp_est.keys()):
            log_var = uncertainty_comp_est['log_var_map']
            weight_map = uncertainty_comp_est['weight_map']
            if isinstance(weight_map, list):
                # if this is list
                proba_map = []
                for weight_map_item in weight_map:
                    proba_map.append(torch.nn.functional.softmax(weight_map_item.detach(), dim=1))
                n = 1 + len(proba_map)
                num_mixture_mode = proba_map[0].shape[1]
            else:
                proba_map = torch.nn.functional.softmax(weight_map.detach(), dim=1)
                n = 2
                num_mixture_mode = proba_map.shape[1]  # number of different modes

            fig, axis = plt.subplots(n, num_mixture_mode * 2 + 2, figsize=(25, 10))
            # uncertainty
            if isinstance(proba_map, list):
                for ind in range(len(proba_map)):
                    for ind_mode in range(num_mixture_mode):
                        proba_map_numpy = proba_map[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                        axis[1 + ind][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                        axis[1 + ind][ind_mode * 2].set_title(
                            "uncertainty_{} \n Var Mode {} Proba map \n min={}, max={}".format(ind, ind_mode,
                                                                                               round(
                                                                                                   proba_map_numpy.min()),
                                                                                               round(
                                                                                                   proba_map_numpy.max())))
                        log_var_numpy = log_var[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                        axis[1 + ind][ind_mode * 2 + 1].imshow(log_var_numpy)
                        axis[1 + ind][ind_mode * 2 + 1].set_title(
                            "uncertainty_{} \n Var Mode {} Log Variance:\n var min={}, max={}"
                                .format(ind, ind_mode, round(np.exp(log_var_numpy).min()),
                                        round(np.exp(log_var_numpy).max())))

                    avg_variance = uncertainty_comp_est['variance'].squeeze().cpu().numpy().astype(np.float32)
                    axis[1 + ind][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
                    axis[1 + ind][num_mixture_mode * 2].set_title(
                        "uncertainty_{} \n variance of the mixture\n min={}, max={}"
                            .format(ind, round(avg_variance.min()), round(avg_variance.max())))

                    probability_interval = uncertainty_comp_est['p_r'].squeeze().cpu().numpy().squeeze()
                    axis[1 + ind][num_mixture_mode * 2 + 1].imshow(probability_interval, vmin=0, vmax=1.0)
                    axis[1 + ind][num_mixture_mode * 2 + 1].set_title(
                        "proba of confidence value within 1\n min={}, max={}".format(
                            round(probability_interval.min() * 100),
                            round(probability_interval.max() * 100)))

            else:
                for ind_mode in range(num_mixture_mode):
                    proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                    axis[1][ind_mode * 2].set_title("Var Mode {} Proba map \n, min={:.2f}, max={:.2f}".format(ind_mode,
                                                                                                      round(
                                                                                                          proba_map_numpy.min(), 2),
                                                                                                      round(
                                                                                                          proba_map_numpy.max(), 2)))
                    log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[1][ind_mode * 2 + 1].imshow(log_var_numpy)
                    axis[1][ind_mode * 2 + 1].set_title("Var Mode {} Log Variance:\n var min={:.2f}, max={:.2f}"
                                                        .format(ind_mode, round(np.exp(log_var_numpy).min(), 2),
                                                                round(np.exp(log_var_numpy).max(), 2)))

                    if plot_log_var:
                        fig_var, ax = plt.subplots()
                        plt.axis('off')
                        ax.imshow(log_var_numpy)
                        fig_var.savefig(os.path.join(save_path, 'individual_variances', "{}_variance_{}.png".format(name_image, ind_mode)),
                                        bbox_inches='tight', pad_inches = 0)
                        plt.close(fig_var)

                        fig_proba, ax = plt.subplots()
                        plt.axis('off')
                        ax.imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                        fig_proba.savefig(os.path.join(save_path, 'individual_variances', "{}_proba_map_{}.png".format(name_image, ind_mode)),
                                        bbox_inches='tight', pad_inches = 0)
                        plt.close(fig_proba)

                avg_variance = uncertainty_comp_est['variance'].squeeze().cpu().numpy().astype(np.float32)
                axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=50.0)
                axis[1][num_mixture_mode * 2].set_title("variance of the mixture\n, min={}, max={}"
                                                        .format(round(avg_variance.min()),
                                                                round(avg_variance.max())))

                probability_interval = uncertainty_comp_est['p_r'].squeeze().cpu().numpy().squeeze()
                axis[1][num_mixture_mode * 2 + 1].imshow(probability_interval, vmin=0, vmax=1.0)
                axis[1][num_mixture_mode * 2 + 1].set_title(
                    "proba of confidence value within 1\n min={}, max={}".format(
                        round(probability_interval.min() * 100),
                        round(probability_interval.max() * 100)))

                if plot_log_var:
                    fig_proba, ax = plt.subplots()
                    plt.axis('off')
                    ax.imshow(probability_interval, vmin=0.0, vmax=1.0)
                    fig_proba.savefig(os.path.join(save_path, 'individual_variances', "{}_proba_R_1.png".format(name_image)),
                                      bbox_inches='tight', pad_inches = 0)
                    plt.close(fig_proba)

                    probability_interval_3 = estimate_probability_of_confidence_interval_of_mixture_density(uncertainty_comp_est,
                                                                                                            R=3.0).cpu().numpy().squeeze()
                    fig_proba, ax = plt.subplots()
                    plt.axis('off')
                    ax.imshow(probability_interval_3, vmin=0.0, vmax=1.0)
                    fig_proba.savefig(os.path.join(save_path, 'individual_variances', "{}_proba_R_3.png".format(name_image)),
                                      bbox_inches='tight', pad_inches = 0)
                    plt.close(fig_proba)


                    probability_interval_5 = estimate_probability_of_confidence_interval_of_mixture_density(uncertainty_comp_est,
                                                                                                            R=5.0).cpu().numpy().squeeze()
                    fig_proba, ax = plt.subplots()
                    plt.axis('off')
                    ax.imshow(probability_interval_5, vmin=0.0, vmax=1.0)
                    fig_proba.savefig(os.path.join(save_path, 'individual_variances', "{}_proba_R_5.png".format(name_image)),
                                      bbox_inches='tight', pad_inches = 0)
                    plt.close(fig_proba)

            axis[0][0].imshow(source_image)
            axis[0][0].set_title("source image")
            axis[0][1].imshow(target_image)
            axis[0][1].set_title("target image")
            axis[0][2].imshow(remapped_est)
            axis[0][2].set_title("source remapped with network")
            axis[0][3].imshow(image_matches_gt)
            axis[0][3].set_title("image matches gt, nbr_matches={}".format(len(Xs)))
            alpha = 0.5
            axis[0][5].imshow((remapped_est * alpha + (1.0 - alpha) * target_image).astype(np.uint8))
            axis[0][5].set_title("source remapped with network overlayed with target")
            if mask_used is not None:
                mask_used = mask_used.squeeze().cpu().numpy()
                axis[0][4].imshow(remapped_est * np.tile(np.expand_dims(mask_used.astype(np.uint8), axis=2), (1, 1, 3)))
                axis[0][4].set_title("source remapped with network")

    fig.savefig('{}/{}.png'.format(save_path, name_image),
                bbox_inches='tight')
    plt.close(fig)


def plot_individual_images(save_path, name_image, source_image, target_image, flow_est,
                           mask_used=None, color=[255, 102, 51]):
    if not isinstance(source_image, np.ndarray):
        source_image = source_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        target_image = target_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    else:
        # numpy array
        if not source_image.shape[2] == 3:
            source_image = source_image.transpose(1, 2, 0)
            target_image = target_image.transpose(1, 2, 0)

    flow_target = flow_est.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    remapped_est = remap_using_flow_fields(source_image, flow_target[:, :, 0], flow_target[:, :, 1])

    max_mapping = 520
    max_flow = 400
    rgb_flow = flow_to_image(flow_target, max_flow)
    rgb_mapping = flow_to_image(convert_flow_to_mapping(flow_target, False), max_mapping)

    if not os.path.isdir(os.path.join(save_path, 'individual_images')):
        os.makedirs(os.path.join(save_path, 'individual_images'))

    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_image_s.png".format(name_image)), source_image)
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_image_t.png".format(name_image)), target_image)
    imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_warped_s.png".format(name_image)),
                    remapped_est)
    # imageio.imwrite(os.path.join(save_path, 'individual_images',  "{}_rgb_flow.png".format(name_image)), rgb_flow)
    # imageio.imwrite(os.path.join(save_path, 'individual_images',  "{}_rgb_mapping.png".format(name_image)),
    # rgb_mapping)
    if mask_used is not None:
        mask_used = mask_used.squeeze().cpu().numpy()
        imageio.imwrite(os.path.join(save_path, 'individual_images', "{}_mask.png".format(name_image)),
                        mask_used.astype(np.uint8) * 255)
        imageio.imwrite(
            os.path.join(save_path, 'individual_images', "{}_image_s_warped_and_mask.png".format(name_image)),
            remapped_est * np.tile(np.expand_dims(mask_used.astype(np.uint8), axis=2), (1, 1, 3)))

        # overlay mask on warped image
        img_mask_overlay_color = overlay_semantic_mask(remapped_est.astype(np.uint8),
                                                       255 - mask_used.astype(np.uint8) * 255, color=color)
        imwrite(os.path.join(save_path, 'individual_images',
                             '{}_warped_overlay_mask_color.png'.format(name_image)), img_mask_overlay_color)

        # overlay warped images in confident regions over target image
        img_warped_overlay_on_target = replace_area(remapped_est,
                                                    mask_used.astype(np.uint8) * 255, target_image, alpha=0.5,
                                                    color=color, thickness=1)
        imwrite(os.path.join(save_path, 'individual_images', '{}_warped_overlay_target.png'.format(name_image)),
                img_warped_overlay_on_target.astype(np.uint8))

