import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.pixel_wise_mapping import warp


def plot_basenet_only_flow(save_path, name, mini_batch, flow_gt, output_net, gt_occ_mask=None,
                           warping_mask=None, occlusion_mask=None, mask_gt=None, normalization=True):
    """
    Plotting of the network predictions (only flow outputs, no uncertainty) as well as the ground-truth flow field
    between the target and source images.
    """

    if warping_mask is not None:
        warping_mask = warping_mask.detach()

    div_flow = 1.0
    b, _, h, w = flow_gt.shape
    fig, axis = plt.subplots(2, 3, figsize=(20, 20))
    if occlusion_mask is not None:
        fig, axis = plt.subplots(2, 4, figsize=(20, 20))
        occlusion_mask = F.adaptive_avg_pool2d(occlusion_mask.detach(), [h, w])
        occlusion_mask = F.sigmoid(occlusion_mask)
    flow_est = F.interpolate(output_net, (h, w), mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x = div_flow * flow_gt.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    if normalization:
        mean_values = torch.tensor([0.485, 0.456, 0.406],
                                   dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        image_1 = (mini_batch['source_image'][0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
        image_2 = (mini_batch['target_image'][0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        image_1 = mini_batch['source_image'][0].cpu().permute(1, 2, 0).numpy()
        image_2 = mini_batch['target_image'][0].cpu().permute(1, 2, 0).numpy()

    remapped_gt = remap_using_flow_fields(image_1, flow_target_x.cpu().numpy(), flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1, flow_est_x.cpu().numpy(), flow_est_y.cpu().numpy())

    axis[0][0].imshow(image_1)
    axis[0][0].set_title("source image")
    axis[0][1].imshow(image_2)
    axis[0][1].set_title("target image")
    if mask_gt is not None:
        mask = mask_gt.cpu().numpy()[0].astype(np.float32)
    else:
        mask = np.ones((h, w))
    axis[0][2].imshow(mask, vmin=0, vmax=1)
    axis[0][2].set_title('mask applied during training')
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[1][0].imshow(remapped_gt)
    axis[1][0].set_title("source remapped with ground truth")
    axis[1][1].imshow(remapped_est)
    axis[1][1].set_title("source remapped with network")
    if warping_mask is not None:
        warping_mask = warping_mask[0].squeeze().cpu().numpy()
        axis[1][2].imshow(warping_mask, vmin=0, vmax=1)
        axis[1][2].set_title("Multiplicative feature mask, \nmin={}, max={}".format(round(warping_mask.min(), 2),
                                                                                  round(warping_mask.max(), 2)))
    if occlusion_mask is not None:
        gt_occlusion_mask = gt_occ_mask.squeeze()[0].cpu().numpy()
        axis[0][3].imshow(gt_occlusion_mask, vmin=0, vmax=1)
        axis[0][3].set_title("ground-truth occlusion mask")
        occlusion_mask = occlusion_mask.squeeze()[0].cpu().numpy()
        axis[1][3].imshow(occlusion_mask, vmin=0, vmax=1)
        axis[1][3].set_title("Estimated occlusion mask, \nmin={}, max={}".format(round(occlusion_mask.min(), 2),
                                                                                  round(occlusion_mask.max(), 2)))
    fig.savefig('{}/{}.jpg'.format(save_path, name),
                bbox_inches='tight')
    plt.close(fig)
    return True


def plot_basenet_during_training(save_path, name, mini_batch, flow_gt, output_net, uncertainty_info=None,
                                 warping_mask=None, mask_gt=None, normalization=True, unimodal_distribution=False,
                                 output_net_bw=None):
    """
    Plotting of the network predictions  as well as the ground-truth flow field between the target and source images.
    """

    if warping_mask is not None:
        warping_mask = warping_mask.detach()
    if uncertainty_info is None:
        return plot_basenet_only_flow(save_path, name,  mini_batch, flow_gt, output_net, mask_gt=mask_gt,
                                      warping_mask=warping_mask)

    b, _, h, w = flow_gt.shape
    div_flow = 1.0
    flow_est = F.interpolate(output_net, (h, w), mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x = div_flow * flow_gt.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    if normalization:
        mean_values = torch.tensor([0.485, 0.456, 0.406],
                                   dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=mini_batch['source_image'].dtype).view(3, 1, 1)
        image_1 = (mini_batch['source_image'][0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
        image_2 = (mini_batch['target_image'][0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        image_1 = mini_batch['source_image'][0].cpu().permute(1, 2, 0).numpy()
        image_2 = mini_batch['target_image'][0].cpu().permute(1, 2, 0).numpy()

    remapped_gt = remap_using_flow_fields(image_1, flow_target_x.cpu().numpy(), flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1, flow_est_x.cpu().numpy(), flow_est_y.cpu().numpy())

    if output_net_bw is not None:
        flow_est_bw = F.interpolate(output_net_bw, (h, w), mode='bilinear', align_corners=False)  # shape Bx2xHxW

        warped_flow_est_bw = - warp(flow_est_bw, flow_est)
        flow_est_bw_x = div_flow * warped_flow_est_bw.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
        flow_est_bw_y = div_flow * warped_flow_est_bw.detach().permute(0, 2, 3, 1)[0, :, :, 1]
        remapped_est_bw = remap_using_flow_fields(image_1, flow_est_bw_x.cpu().numpy(), flow_est_bw_y.cpu().numpy())

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
            num_mixture_mode = proba_map.shape[1] # number of different modes
        fig, axis = plt.subplots(n, num_mixture_mode*2 + 1, figsize=(20, 20))
        axis[0][0].imshow(image_1)
        axis[0][0].set_title("source image")
        axis[0][1].imshow(image_2)
        axis[0][1].set_title("target image")
        if mask_gt is not None:
            mask = mask_gt.cpu().numpy()[0].astype(np.float32)
        else:
            mask = np.ones((h, w))
        axis[0][2].imshow(mask, vmin=0, vmax=1)
        axis[0][2].set_title('mask applied during training')
        axis[0][3].imshow(remapped_gt)
        axis[0][3].set_title("source remapped with ground truth")
        axis[0][4].imshow(remapped_est)
        axis[0][4].set_title("source remapped with network")
        if output_net_bw is not None:
            axis[0][5].imshow(remapped_est_bw)
            axis[0][5].set_title("source remapped with backward flow")

        # uncertainty
        if isinstance(proba_map, list):
            for ind in range(len(proba_map)):
                for ind_mode in range(num_mixture_mode):
                    proba_map_numpy = proba_map[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[1+ind][ind_mode*2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                    axis[1+ind][ind_mode*2].set_title(
                        "uncertainty_{} \n Var Mode {} Proba map \n min={}, max={}".format(ind, ind_mode, round(proba_map_numpy.min(), 2),
                                                                                  round(proba_map_numpy.max(), 2)))
                    log_var_numpy = log_var[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[1+ind][ind_mode*2+1].imshow(log_var_numpy)
                    axis[1+ind][ind_mode*2+1].set_title("uncertainty_{} \n Var Mode {} Log Variance:\n var min={}, max={}"
                                                      .format(ind, ind_mode, round(np.exp(log_var_numpy).min()),
                                                              round(np.exp(log_var_numpy).max())))

                avg_variance = torch.sum(proba_map[ind] * torch.exp(log_var[ind]), dim=1, keepdim=True).squeeze()[0].cpu().numpy().astype(np.float32)
                axis[1+ind][num_mixture_mode*2].imshow(avg_variance, vmin=1.0, vmax=10.0)
                axis[1+ind][num_mixture_mode*2].set_title("uncertainty_{} \n variance of the mixture\n min={}, max={}"
                                                          .format(ind, round(avg_variance.min()), round(avg_variance.max())))

        else:
            for ind_mode in range(num_mixture_mode):
                proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1][ind_mode*2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                axis[1][ind_mode*2].set_title("Var Mode {} Proba map \n, min={}, max={}".format(ind_mode,
                                                                                        round(proba_map_numpy.min(), 2),
                                                                                        round(proba_map_numpy.max(), 2)))
                log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1][ind_mode*2 + 1].imshow(log_var_numpy)
                axis[1][ind_mode*2 + 1].set_title("Var Mode {} Log Variance:\n var min={}, max={}"
                                                      .format(ind_mode, round(np.exp(log_var_numpy).min(), 2),
                                                              round(np.exp(log_var_numpy).max(), 2)))

            avg_variance = torch.sum(proba_map * torch.exp(log_var), dim=1, keepdim=True).squeeze()[
                0].cpu().numpy().astype(np.float32)
            axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
            axis[1][num_mixture_mode * 2].set_title("variance of the mixture\n, min={}, max={}"
                                                          .format(round(avg_variance.min()),
                                                                  round(avg_variance.max())))

        fig.tight_layout()
        fig.savefig('{}/{}.jpg'.format(save_path, name))
        plt.close(fig)
    else:
        if warping_mask is not None:
            fig, axis = plt.subplots(2, 4, figsize=(20, 20))
            axis[1][3].imshow(warping_mask[0].squeeze().cpu().numpy(), vmin=0, vmax=1)
            axis[1][3].set_title("Multiplicative feature mask, min={}, max={}".format(round(warping_mask.min(), 2),
                                                                                      round(warping_mask.max(), 2)))
        else:
            fig, axis = plt.subplots(2, 3, figsize=(20, 20))
        axis[0][0].imshow(image_1)
        axis[0][0].set_title("source image")
        axis[0][1].imshow(image_2)
        axis[0][1].set_title("target image")
        if mask_gt is not None:
            mask = mask_gt.cpu().numpy()[0].astype(np.float32)
        else:
            mask = np.ones((h, w))
        axis[0][2].imshow(mask, vmin=0, vmax=1)
        axis[0][2].set_title('mask applied during training')
        axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
        axis[1][0].imshow(remapped_gt)
        axis[1][0].set_title("source remapped with ground truth")
        axis[1][1].imshow(remapped_est)
        axis[1][1].set_title("source remapped with network")
        log_var_numpy = uncertainty_info.squeeze().cpu().detach().numpy()[0].astype(np.float32)
        axis[1][2].imshow(log_var_numpy, vmin=1.0, vmax= 10.0)
        axis[1][2].set_title("Log Variance\n var min={}, max={}".format(round(np.exp(log_var_numpy).min()),
                                                                        round(np.exp(log_var_numpy).max())))

        fig.tight_layout()
        fig.savefig('{}/{}.jpg'.format(save_path, name))
        plt.close(fig)

    return True


