import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.visualization_utils import draw_matches


def plot_matches(save_path, name, mini_batch, Ks, Kt, predicted_Kt=None, predicted_Ks=None, normalization=True):

    if normalization:
        mean_values = torch.tensor([0.485, 0.456, 0.406],
                                   dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        source_image = (mini_batch['source_image'][0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
        target_image = (mini_batch['target_image'][0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        source_image = mini_batch['source_image'][0].cpu().permute(1, 2, 0).numpy()
        target_image = mini_batch['target_image'][0].cpu().permute(1, 2, 0).numpy()

    Ks = Ks.cpu().numpy().copy()
    Kt = Kt.cpu().numpy().copy()
    image_matches_gt = np.clip(draw_matches(source_image, target_image, Ks, Kt), 0, 1)

    if predicted_Kt is not None:
        predicted_Kt = predicted_Kt.cpu().numpy().copy()
        image_matches_est = np.clip(draw_matches(source_image, target_image, Ks, predicted_Kt), 0, 1)
    elif predicted_Ks is not None:
        predicted_Ks = predicted_Ks.cpu().numpy().copy()
        image_matches_est = np.clip(draw_matches(source_image, target_image, predicted_Ks, Kt), 0, 1)
    else:
        raise ValueError('Both your predicted keypoints are None')
    fig, axis = plt.subplots(1, 2, figsize=(20, 20))
    axis[0].imshow(image_matches_gt, vmin=0, vmax=1.0)
    axis[0].set_title("gt matches")
    axis[1].imshow(image_matches_est, vmin=0, vmax=1.0)
    axis[1].set_title("est matches")
    fig.savefig('{}/{}.jpg'.format(save_path, name),
                bbox_inches='tight')
    plt.close(fig)


def plot_sparse_keypoints(save_path, name, mini_batch, flow_gt, output_net, gt_occ_mask=None,
                          warping_mask=None, occlusion_mask=None, normalization=True, uncertainty_info=None,
                          unimodal_distribution=False, source_image=None, target_image=None):

    b, _, h, w = flow_gt.shape

    flow_est = F.interpolate(output_net, (h, w), mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0

    if source_image is None:
        source_image = mini_batch['source_image']
    if target_image is None:
        target_image = mini_batch['target_image']

    if normalization:
        mean_values = torch.tensor([0.485, 0.456, 0.406],
                                   dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        source_image = (source_image[0].cpu() * std_values +
                        mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
        target_image = (target_image[0].cpu() * std_values +
                        mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        source_image = source_image[0].cpu().permute(1, 2, 0).numpy()
        target_image = target_image[0].cpu().permute(1, 2, 0).numpy()

    flow_gt = flow_gt.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    kp2_where = np.where(np.linalg.norm(flow_gt, axis=-1) > 0)
    n = kp2_where[0].shape[0]
    kp2 = np.concatenate([kp2_where[1].reshape(n, 1), kp2_where[0].reshape(n, 1)], axis=1)
    kp1 = flow_gt[kp2[:, 1], kp2[:, 0]] + kp2
    image_matches_gt = np.clip(draw_matches(source_image, target_image, np.int32(kp1), np.int32(kp2)), 0, 1)

    flow_target = flow_est.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    kp1_estimated = flow_target[kp2[:, 1], kp2[:, 0]] + kp2
    image_matches_estimated = np.clip(draw_matches(source_image, target_image, np.int32(kp1_estimated), np.int32(kp2)), 0, 1)

    remapped_est = np.clip(remap_using_flow_fields(source_image, flow_target[:, :, 0], flow_target[:, :, 1]), 0, 1)

    if warping_mask is None and occlusion_mask is None and uncertainty_info is None:
        fig, axis = plt.subplots(1, 5, figsize = (20, 20))
        axis[0].imshow(source_image, vmin=0, vmax=1.0)
        axis[0].set_title("source image")
        axis[1].imshow(target_image, vmin=0, vmax=1.0)
        axis[1].set_title("target image")
        axis[2].imshow(remapped_est, vmin=0, vmax=1.0)
        axis[2].set_title("source remapped with network")
        axis[3].imshow(image_matches_gt.astype(np.float32), vmin=0, vmax=1)
        axis[3].set_title("image matches gt, nbr_matches={}".format(kp2.shape[0]))
        axis[4].imshow(image_matches_estimated.astype(np.float32), vmin=0, vmax=1)
        axis[4].set_title("image matches est, nbr_matches={}".format(kp2.shape[0]))
    else:
        if warping_mask is not None:
            fig, axis = plt.subplots(3, 5, figsize=(20, 20))
            warping_mask = F.adaptive_avg_pool2d(warping_mask.detach(), [h, w])
            warping_mask = F.sigmoid(warping_mask)
        if occlusion_mask is not None:
            fig, axis = plt.subplots(3, 5, figsize=(20, 20))
            occlusion_mask = F.adaptive_avg_pool2d(occlusion_mask.detach(), [h, w])
            occlusion_mask = F.sigmoid(occlusion_mask)
        elif uncertainty_info is not None:
            if not unimodal_distribution:
                log_var = uncertainty_info[0].detach()
                weight_map = uncertainty_info[1].detach()
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
                fig, axis = plt.subplots(n, num_mixture_mode * 2 + 1, figsize=(20, 20))

                if isinstance(proba_map, list):
                    for ind in range(len(proba_map)):
                        for ind_mode in range(num_mixture_mode):
                            proba_map_numpy = proba_map[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                            axis[1 + ind][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                            axis[1 + ind][ind_mode * 2].set_title(
                                "uncertainty_{} \n Var Mode {} Proba map \n min={}, max={}".format(ind, ind_mode, round(
                                    proba_map_numpy.min()),
                                                                                                   round(
                                                                                                       proba_map_numpy.max())))
                            log_var_numpy = log_var[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                            axis[1 + ind][ind_mode * 2 + 1].imshow(log_var_numpy)
                            axis[1 + ind][ind_mode * 2 + 1].set_title(
                                "uncertainty_{} \n Var Mode {} Log Variance:\n var min={}, max={}"
                                .format(ind, ind_mode, round(np.exp(log_var_numpy).min()),
                                        round(np.exp(log_var_numpy).max())))

                        avg_variance = torch.sum(proba_map[ind] * torch.exp(log_var[ind]), dim=1, keepdim=True).squeeze()[
                            0].cpu().numpy().astype(np.float32)
                        axis[1 + ind][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
                        axis[1 + ind][num_mixture_mode * 2].set_title(
                            "uncertainty_{} \n variance of the mixture\n min={}, max={}"
                            .format(ind, round(avg_variance.min()), round(avg_variance.max())))

                else:
                    for ind_mode in range(num_mixture_mode):
                        proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                        axis[1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                        axis[1][ind_mode * 2].set_title("Var Mode {} Proba map \n, min={}, max={}".format(ind_mode,
                                                                                                          round(
                                                                                                              proba_map_numpy.min()),
                                                                                                          round(
                                                                                                              proba_map_numpy.max())))
                        log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                        axis[1][ind_mode * 2 + 1].imshow(log_var_numpy)
                        axis[1][ind_mode * 2 + 1].set_title("Var Mode {} Log Variance:\n var min={}, max={}"
                                                            .format(ind_mode, round(np.exp(log_var_numpy).min()),
                                                                    round(np.exp(log_var_numpy).max())))

                    avg_variance = torch.sum(proba_map * torch.exp(log_var), dim=1, keepdim=True).squeeze()[
                        0].cpu().numpy().astype(np.float32)
                    axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
                    axis[1][num_mixture_mode * 2].set_title("variance of the mixture\n, min={}, max={}"
                                                            .format(round(avg_variance.min()),
                                                                    round(avg_variance.max())))

            else:
                fig, axis = plt.subplots(1, 6, figsize=(20, 20))
                log_var = uncertainty_info.squeeze().cpu().detach().numpy().astype(np.float32)
                axis[0][5].imshow(log_var)
                axis[0][5].set_title('log variance. \n var min is {}, max {}'.format(round(np.exp(log_var).min()),
                                                                                     round(np.exp(log_var).max())))

        else:
            raise NotImplementedError

        axis[0][0].imshow(source_image, vmin=0, vmax=1.0)
        axis[0][0].set_title("source image")
        axis[0][1].imshow(target_image, vmin=0, vmax=1.0)
        axis[0][1].set_title("target image")
        axis[0][2].imshow(remapped_est, vmin=0, vmax=1.0)
        axis[0][2].set_title("source remapped with network")

        axis[0][4].imshow(image_matches_gt.astype(np.float32), vmin=0, vmax=1)
        axis[0][4].set_title("image matches gt, nbr_matches={}".format(kp2.shape[0]))

        if warping_mask is not None:
            warping_mask = warping_mask.squeeze()[0].cpu().numpy()
            axis[2][0].imshow(warping_mask, vmin=0, vmax=1)
            axis[2][0].set_title("Multiplicative feature mask, \nmin={}, max={}".format(round(warping_mask.min(), 2),
                                                                                        round(warping_mask.max(), 2)))
        if occlusion_mask is not None:
            gt_occlusion_mask = gt_occ_mask.squeeze()[0].cpu().numpy()
            axis[2][1].imshow(gt_occlusion_mask, vmin=0, vmax=1)
            axis[2][1].set_title("ground-truth occlusion mask")
            occlusion_mask = occlusion_mask.squeeze()[0].cpu().numpy()
            axis[2][2].imshow(occlusion_mask, vmin=0, vmax=1)
            axis[2][2].set_title("Estimated occlusion mask, \nmin={}, max={}".format(round(occlusion_mask.min(), 2),
                                                                                     round(occlusion_mask.max(), 2)))
    fig.savefig('{}/{}.jpg'.format(save_path, name),
                bbox_inches='tight')
    plt.close(fig)
    return True