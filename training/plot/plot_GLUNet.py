import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.visualization_utils import draw_matches


def plot_predictions(save_path, epoch, batch, source_image, target_image, source_image_256, target_image_256,
                     output_net,  output_net_256, mask=None, mask_256=None, div_flow=1.0):
    """
    Plotting of the network predictions, when no ground-truth flow is known between the images.
    For GLU-Net, two sets of predictions, at two resolutions.
    """

    b, _, h_original, w_original = source_image.shape
    b, _, h_256, w_256 = source_image_256.shape
    # resolution original
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())

    # resolution 256x256
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
                                 mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]

    image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
                                               flow_est_y_256.cpu().numpy())

    fig, axis = plt.subplots(2, 4, figsize=(20, 20))
    axis[0][0].imshow(image_1.numpy())
    axis[0][0].set_title("original reso: \nsource image")
    axis[0][1].imshow(image_2.numpy())
    axis[0][1].set_title("original reso: \ntarget image")
    if mask is not None:
        mask = mask.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask = np.ones((h_original, w_original))
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[0][3].imshow(remapped_est)
    axis[0][3].set_title("original reso: \nsource remapped with network")

    axis[1][0].imshow(image_1_256.numpy())
    axis[1][0].set_title("reso 256: \nsource image")
    axis[1][1].imshow(image_2_256.numpy())
    axis[1][1].set_title("reso 256:\ntarget image")
    if mask_256 is not None:
        mask_256 = mask_256.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask_256 = np.ones((h_256, w_256))
    axis[1][2].imshow(mask_256, vmin=0.0, vmax=1.0)
    axis[1][3].imshow(remapped_est_256)
    axis[1][3].set_title("reso 256: \nsource remapped with network")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)
    return True


def plot_during_training(save_path, epoch, batch, source_image, target_image, source_image_256, target_image_256,
                         flow_gt_original, flow_gt_256, output_net,  output_net_256, mask=None, mask_256=None,
                         div_flow=1.0):
    """
    Plotting of the network predictions (only flow outputs, no uncertainty) as well as the ground-truth flow field
    between the target and source images. For GLU-Net, two sets of predictions, at two resolutions.
    """
    b, _, h_original, w_original = flow_gt_original.shape
    b, _, h_256, w_256 = flow_gt_256.shape
    # resolution original
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_target_x = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_x.cpu().numpy(),
                                          flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())

    # resolution 256x256
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
                                 mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x_256.shape == flow_target_x_256.shape

    image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt_256 = remap_using_flow_fields(image_1_256.numpy(),
                                              flow_target_x_256.cpu().numpy(),
                                              flow_target_y_256.cpu().numpy())
    remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
                                               flow_est_y_256.cpu().numpy())

    fig, axis = plt.subplots(2, 5, figsize=(20, 20))
    axis[0][0].imshow(image_1.numpy())
    axis[0][0].set_title("original reso: \nsource image")
    axis[0][1].imshow(image_2.numpy())
    axis[0][1].set_title("original reso: \ntarget image")
    if mask is not None:
        mask = mask.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask = np.ones((h_original, w_original))
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[0][2].set_title("original reso: \nmask applied during training")
    axis[0][3].imshow(remapped_gt)
    axis[0][3].set_title("original reso : \nsource remapped with ground truth")
    axis[0][4].imshow(remapped_est)
    axis[0][4].set_title("original reso: \nsource remapped with network")
    axis[1][0].imshow(image_1_256.numpy())
    axis[1][0].set_title("reso 256: \nsource image")
    axis[1][1].imshow(image_2_256.numpy())
    axis[1][1].set_title("reso 256:\ntarget image")
    if mask_256 is not None:
        mask_256 = mask_256.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask_256 = np.ones((h_256, w_256))
    axis[1][2].imshow(mask_256, vmin=0.0, vmax=1.0)
    axis[1][2].set_title("reso 256: \nmask applied during training")
    axis[1][3].imshow(remapped_gt_256)
    axis[1][3].set_title("reso 256: \nsource remapped with ground truth")
    axis[1][4].imshow(remapped_est_256)
    axis[1][4].set_title("reso 256: \nsource remapped with network")
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch),
                bbox_inches='tight')
    plt.close(fig)
    return True


def plot_during_training_with_uncertainty(save_path, epoch, batch, source_image, target_image, source_image_256,
                                          target_image_256, flow_gt_original, flow_gt_256, output_net, output_net_256,
                                          uncertainty_info_original, uncertainty_info_256, mask=None,
                                          mask_256=None, div_flow=1.0):
    """
    Plotting of the network predictions  as well as the ground-truth flow field between the target and source images.
    For GLU-Net, two sets of predictions, at two resolutions.
    """

    if uncertainty_info_original is None:
        return plot_during_training(save_path, epoch, batch, source_image, target_image, source_image_256, target_image_256,
                                    flow_gt_original, flow_gt_256, output_net, output_net_256, mask=mask, mask_256=mask_256,
                                    div_flow=div_flow)
    # resolution original
    b, _, h_original, w_original = flow_gt_original.shape
    b, _, h_256, w_256 = flow_gt_256.shape
    flow_est_original = F.interpolate(output_net, (h_original, w_original),
                                      mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_target_x = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y = div_flow * flow_gt_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y = div_flow * flow_est_original.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x.shape == flow_target_x.shape

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)
    image_1 = (source_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2 = (target_image.detach()[0].cpu() * std_values +
               mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt = remap_using_flow_fields(image_1.numpy(),
                                          flow_target_x.cpu().numpy(),
                                          flow_target_y.cpu().numpy())
    remapped_est = remap_using_flow_fields(image_1.numpy(), flow_est_x.cpu().numpy(),
                                           flow_est_y.cpu().numpy())

    # resolution 256x256
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256),
                                 mode='bilinear', align_corners=False)  # shape Bx2xHxW
    # for batch 0
    flow_target_x_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape HxW
    flow_target_y_256 = div_flow * flow_gt_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    flow_est_x_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 0]  # shape BxHxW
    flow_est_y_256 = div_flow * flow_est_256.detach().permute(0, 2, 3, 1)[0, :, :, 1]
    assert flow_est_x_256.shape == flow_target_x_256.shape

    image_1_256 = (source_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    image_2_256 = (target_image_256.detach()[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0)
    remapped_gt_256 = remap_using_flow_fields(image_1_256.numpy(),
                                              flow_target_x_256.cpu().numpy(),
                                              flow_target_y_256.cpu().numpy())
    remapped_est_256 = remap_using_flow_fields(image_1_256.numpy(), flow_est_x_256.cpu().numpy(),
                                               flow_est_y_256.cpu().numpy())

    # uncertainty stuff
    log_var = uncertainty_info_original[0]
    weight_map = uncertainty_info_original[1]

    log_var_256 = uncertainty_info_256[0]
    weight_map_256 = uncertainty_info_256[1]
    if isinstance(weight_map, list):
        # if this is list
        proba_map = []
        proba_map_256 = []
        for item, weight_map_item in enumerate(weight_map):
            proba_map.append(torch.nn.functional.softmax(weight_map_item.detach(), dim=1))
            proba_map_256.append(torch.nn.functional.softmax(weight_map_256[item].detach(), dim=1))
        n = 2 + len(proba_map) * 2
        n1 = 1 + len(proba_map)
        num_mixture_mode = proba_map[0].shape[1]
    else:
        proba_map = torch.nn.functional.softmax(weight_map.detach(), dim=1)
        proba_map_256 = torch.nn.functional.softmax(weight_map_256.detach(), dim=1)
        log_var_256 = log_var_256.detach()
        log_var = log_var.detach()
        n = 2 + 2
        n1 = 2
        num_mixture_mode = proba_map.shape[1]
    fig, axis = plt.subplots(n, num_mixture_mode*2 + 1, figsize=(20, 20))

    axis[0][0].imshow(image_1.numpy())
    axis[0][0].set_title("original reso: \nsource image")
    axis[0][1].imshow(image_2.numpy())
    axis[0][1].set_title("original reso: \ntarget image")
    if mask is not None:
        mask = mask.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask = np.ones((h_original, w_original))
    axis[0][2].imshow(mask, vmin=0.0, vmax=1.0)
    axis[0][2].set_title("original reso: \nmask applied during training")
    axis[0][3].imshow(remapped_gt)
    axis[0][3].set_title("original reso : \nsource remapped with ground truth")
    axis[0][4].imshow(remapped_est)
    axis[0][4].set_title("original reso: \nsource remapped with network")

    if isinstance(proba_map, list):
        for ind in range(len(proba_map)):
            for ind_mode in range(num_mixture_mode):
                proba_map_numpy = proba_map[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1 + ind][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                axis[1 + ind][ind_mode * 2].set_title(
                    "uncertainty_{} \n Mixture component {} Proba map \n min={:.2f}, max={:.2f}"
                        .format(ind, ind_mode, proba_map_numpy.min(), proba_map_numpy.max()))
                log_var_numpy = log_var[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1 + ind][ind_mode * 2 + 1].imshow(log_var_numpy)
                axis[1 + ind][ind_mode * 2 + 1].set_title(
                    "uncertainty_{} \n Mixture component {} Log Variance:\n var min={:.2f}, max={:.2f}"
                    .format(ind, ind_mode, np.exp(log_var_numpy).min(), np.exp(log_var_numpy).max()))

            avg_variance = torch.sum(proba_map[ind].detach() * torch.exp(log_var[ind].detach()), dim=1, keepdim=True).squeeze()[
                0].cpu().numpy().astype(np.float32)
            axis[1 + ind][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
            axis[1 + ind][num_mixture_mode * 2].set_title("uncertainty_{} \n variance of the mixture\n min={:.2f}, max={:.2f}"
                                                          .format(ind, avg_variance.min(), avg_variance.max()))

    else:
        for ind_mode in range(num_mixture_mode):
            proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
            axis[1][ind_mode * 2].set_title("Mixture component {} Proba map \n, min={:.2f}, max={:.2f}"
                                            .format(ind_mode, proba_map_numpy.min(), proba_map_numpy.max()))
            log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[1][ind_mode * 2 + 1].imshow(log_var_numpy)
            axis[1][ind_mode * 2 + 1].set_title("Mixture component {} Log Variance:\n var min={:.2f}, max={:.2f}"
                                                .format(ind_mode, np.exp(log_var_numpy).min(),
                                                        np.exp(log_var_numpy).max()))

        avg_variance = torch.sum(proba_map * torch.exp(log_var), dim=1, keepdim=True).squeeze()[
            0].cpu().numpy().astype(np.float32)
        axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
        axis[1][num_mixture_mode * 2].set_title("variance of the mixture\n min={:.2f}, max={:.2f}"
                                                .format(avg_variance.min(), avg_variance.max()))

    axis[n1][0].imshow(image_1_256.numpy())
    axis[n1][0].set_title("reso 256: \nsource image")
    axis[n1][1].imshow(image_2_256.numpy())
    axis[n1][1].set_title("reso 256:\ntarget image")
    if mask_256 is not None:
        mask_256 = mask_256.detach()[0].cpu().numpy().astype(np.float32)
    else:
        mask_256 = np.ones((h_256, w_256))
    axis[n1][2].imshow(mask_256, vmin=0.0, vmax=1.0)
    axis[n1][2].set_title("reso 256: \nmask applied during training")
    axis[n1][3].imshow(remapped_gt_256)
    axis[n1][3].set_title("reso 256: \nsource remapped with ground truth")
    axis[n1][4].imshow(remapped_est_256)
    axis[n1][4].set_title("reso 256: \nsource remapped with network")

    if isinstance(proba_map_256, list):
        for ind in range(len(proba_map_256)):
            for ind_mode in range(num_mixture_mode):
                proba_map_numpy = proba_map_256[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[n1+1 + ind][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                axis[n1+1 + ind][ind_mode * 2].set_title(
                    "uncertainty_{} \n Mixture component {} Proba map \n min={:.2f}, max={:.2f}"
                        .format(ind, ind_mode, proba_map_numpy.min(), proba_map_numpy.max()))
                log_var_numpy = log_var_256[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[n1+1 + ind][ind_mode * 2 + 1].imshow(log_var_numpy)
                axis[n1+1 + ind][ind_mode * 2 + 1].set_title(
                    "uncertainty_{} \n Mixture component {} Log Variance:\n var min={:.2f}, max={:.2f}"
                    .format(ind, ind_mode, np.exp(log_var_numpy).min(), np.exp(log_var_numpy).max()))

            avg_variance = torch.sum(proba_map_256[ind].detach() * torch.exp(log_var_256[ind].detach()), dim=1, keepdim=True).squeeze()[
                0].cpu().numpy().astype(np.float32)
            axis[n1+1 + ind][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
            axis[n1+1 + ind][num_mixture_mode * 2].set_title("uncertainty_{} \n variance of the mixture\n min={:.2f}, max={:.2f}"
                                                          .format(ind, avg_variance.min(), avg_variance.max()))

    else:
        for ind_mode in range(num_mixture_mode):
            proba_map_numpy = proba_map_256[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[n1+1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
            axis[n1+1][ind_mode * 2].set_title("Mixture component {} Proba map \n, min={:.2f}, max={:.2f}"
                                               .format(ind_mode, proba_map_numpy.min(), proba_map_numpy.max()))
            log_var_numpy = log_var_256[0, ind_mode].cpu().detach().numpy().astype(np.float32)
            axis[n1+1][ind_mode * 2 + 1].imshow(log_var_numpy)
            axis[n1+1][ind_mode * 2 + 1].set_title("Mixture component {} Log Variance:\n var min={:.2f}, max={:.2f}"
                                                .format(ind_mode, np.exp(log_var_numpy).min(), np.exp(log_var_numpy).max()))

        avg_variance = torch.sum(proba_map_256 * torch.exp(log_var_256), dim=1, keepdim=True).squeeze()[
            0].cpu().numpy().astype(np.float32)
        axis[n1+1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
        axis[n1+1][num_mixture_mode * 2].set_title("Variance of the mixture:\n min={:.2f}, max={:.2f}"
                                                .format(avg_variance.min(), avg_variance.max()))

    fig.tight_layout()
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch))
    plt.close(fig)
    return True


def plot_sparse_keypoints_GLUNet(save_path, epoch, batch, source_image, target_image, source_image_256, target_image_256,
                                 flow_gt, flow_gt_256, output_net, output_net_256, uncertainty_info_original=None,
                                 uncertainty_info_256=None, normalization=True):
    # for batch 0

    if normalization:
        mean_values = torch.tensor([0.485, 0.456, 0.406],
                                   dtype=source_image.dtype).view(3, 1, 1)
        std_values = torch.tensor([0.229, 0.224, 0.225],
                                  dtype=source_image.dtype).view(3, 1, 1)
        image_1 = (source_image[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
        image_2 = (target_image[0].cpu() * std_values +
                       mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
        image_1_256 = (source_image_256[0].cpu() * std_values +
                   mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
        image_2_256 = (target_image_256[0].cpu() * std_values +
                       mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    else:
        image_1 = source_image[0].cpu().permute(1, 2, 0).numpy()
        image_2 = target_image[0].cpu().permute(1, 2, 0).numpy()
        image_1_256 = source_image_256[0].cpu().permute(1, 2, 0).numpy()
        image_2_256 = target_image_256[0].cpu().permute(1, 2, 0).numpy()

    # at original resolution
    h, w, _ = image_1.shape
    flow_est = F.interpolate(output_net, (h, w), mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_gt = flow_gt.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    kp2_where = np.where(np.abs(flow_gt) > 0)
    n = kp2_where[0].shape[0]
    kp2 = np.concatenate([kp2_where[1].reshape(n, 1), kp2_where[0].reshape(n, 1)], axis=1)
    kp1 = flow_gt[kp2[:, 1], kp2[:, 0]] + kp2
    image_matches_gt = np.clip(draw_matches(image_1, image_2, np.int32(kp1), np.int32(kp2)), 0, 1)

    flow_target = flow_est.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    kp1_estimated = flow_target[kp2[:, 1], kp2[:, 0]] + kp2
    image_matches_estimated = np.clip(draw_matches(image_1, image_2, np.int32(kp1_estimated), np.int32(kp2)), 0, 1)

    remapped_est = np.clip(remap_using_flow_fields(image_1, flow_target[:,:,0], flow_target[:,:,1]), 0, 1)

    # at 256x256 resolution
    h_256, w_256, _ = image_1_256.shape
    flow_est_256 = F.interpolate(output_net_256, (h_256, w_256), mode='bilinear', align_corners=False)  # shape Bx2xHxW
    flow_gt_256 = flow_gt_256.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    kp2_where_256 = np.where(np.abs(flow_gt_256) > 0)
    n = kp2_where_256[0].shape[0]
    kp2_256 = np.concatenate([kp2_where_256[1].reshape(n, 1), kp2_where_256[0].reshape(n, 1)], axis=1)
    kp1_256 = flow_gt_256[kp2_256[:, 1], kp2_256[:, 0]] + kp2_256
    image_matches_gt_256 = np.clip(draw_matches(image_1_256, image_2_256, np.int32(kp1_256), np.int32(kp2_256)), 0, 1)

    flow_target_256 = flow_est_256.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
    kp1_estimated_256 = flow_target_256[kp2_256[:, 1], kp2_256[:, 0]] + kp2_256
    image_matches_estimated_256 = np.clip(draw_matches(image_1_256, image_2_256, np.int32(kp1_estimated_256), np.int32(kp2_256)), 0, 1)

    remapped_est_256 = np.clip(remap_using_flow_fields(image_1_256, flow_target_256[:,:,0], flow_target_256[:,:,1]), 0, 1)

    # uncertainty stuff
    if uncertainty_info_original is not None:
        # uncertainty stuff
        log_var = uncertainty_info_original[0]
        weight_map = uncertainty_info_original[1]

        log_var_256 = uncertainty_info_256[0]
        weight_map_256 = uncertainty_info_256[1]
        if isinstance(weight_map, list):
            # if this is list
            proba_map = []
            proba_map_256 = []
            for item, weight_map_item in enumerate(weight_map):
                proba_map.append(torch.nn.functional.softmax(weight_map_item.detach(), dim=1))
                proba_map_256.append(torch.nn.functional.softmax(weight_map_256[item].detach(), dim=1))
            n = 2 + len(proba_map) * 2
            n1 = 1 + len(proba_map)
            num_mixture_mode = proba_map[0].shape[1]
        else:
            proba_map = torch.nn.functional.softmax(weight_map.detach(), dim=1)
            proba_map_256 = torch.nn.functional.softmax(weight_map_256.detach(), dim=1)
            log_var = log_var.detach()
            log_var_256 = log_var_256.detach()
            n = 2 + 2
            n1 = 2
            num_mixture_mode = proba_map.shape[1]

    else:
        n = 2
        num_mixture_mode = 2

    fig, axis = plt.subplots(n, num_mixture_mode * 2 + 1, figsize=(20, 20))
    axis[0][0].imshow(image_1, vmin=0, vmax=1.0)
    axis[0][0].set_title("source image")
    axis[0][1].imshow(image_2, vmin=0, vmax=1.0)
    axis[0][1].set_title("target image")
    axis[0][2].imshow(remapped_est, vmin=0, vmax=1.0)
    axis[0][2].set_title("source remapped with network")

    axis[0][3].imshow(image_matches_gt.astype(np.float32), vmin=0, vmax=1)
    axis[0][3].set_title("image matches gt, nbr_matches={}".format(kp2.shape[0]))
    axis[0][4].imshow(image_matches_estimated.astype(np.float32), vmin=0, vmax=1)
    axis[0][4].set_title("image matches estimated")

    if uncertainty_info_original is None:
        axis[1][0].imshow(image_1_256, vmin=0, vmax=1.0)
        axis[1][0].set_title("source image 256")
        axis[1][1].imshow(image_2_256, vmin=0, vmax=1.0)
        axis[1][1].set_title("target image 256")
        axis[1][2].imshow(remapped_est_256, vmin=0, vmax=1.0)
        axis[1][2].set_title("source remapped with network 256")

        axis[1][3].imshow(image_matches_gt_256.astype(np.float32), vmin=0, vmax=1)
        axis[1][3].set_title("image matches gt 256, nbr_matches={}".format(kp2_256.shape[0]))
        axis[1][4].imshow(image_matches_estimated_256.astype(np.float32), vmin=0, vmax=1)
        axis[1][4].set_title("image matches estimated 256")

    else:
        if isinstance(proba_map, list):
            for ind in range(len(proba_map)):
                for ind_mode in range(num_mixture_mode):
                    proba_map_numpy = proba_map[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[1 + ind][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                    axis[1 + ind][ind_mode * 2].set_title(
                        "uncertainty_{} \n Mixture component {} Proba map \n min={}, max={}".format(ind, ind_mode,
                                                                                           round(proba_map_numpy.min()),
                                                                                           round(
                                                                                               proba_map_numpy.max())))
                    log_var_numpy = log_var[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[1 + ind][ind_mode * 2 + 1].imshow(log_var_numpy)
                    axis[1 + ind][ind_mode * 2 + 1].set_title(
                        "uncertainty_{} \n Mixture component {} Log Variance:\n var min={}, max={}"
                            .format(ind, ind_mode, round(np.exp(log_var_numpy).min()),
                                    round(np.exp(log_var_numpy).max())))

                avg_variance = torch.sum(proba_map[ind].detach() * torch.exp(log_var[ind].detach()), dim=1, keepdim=True).squeeze()[
                    0].cpu().numpy().astype(np.float32)
                axis[1 + ind][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
                axis[1 + ind][num_mixture_mode * 2].set_title(
                    "uncertainty_{} \n variance of the mixture\n min={}, max={}"
                    .format(ind, round(avg_variance.min()),
                            round(avg_variance.max())))

        else:
            for ind_mode in range(num_mixture_mode):
                proba_map_numpy = proba_map[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                axis[1][ind_mode * 2].set_title("Mixture component {} Proba map \n, min={}, max={}".format(ind_mode,
                                                                                                  round(
                                                                                                      proba_map_numpy.min()),
                                                                                                  round(
                                                                                                      proba_map_numpy.max())))
                log_var_numpy = log_var[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[1][ind_mode * 2 + 1].imshow(log_var_numpy)
                axis[1][ind_mode * 2 + 1].set_title("Mixture component {} Log Variance:\n var min={}, max={}"
                                                    .format(ind_mode, round(np.exp(log_var_numpy).min()),
                                                            round(np.exp(log_var_numpy).max())))

            avg_variance = torch.sum(proba_map * torch.exp(log_var), dim=1, keepdim=True).squeeze()[
                0].cpu().numpy().astype(np.float32)
            axis[1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
            axis[1][num_mixture_mode * 2].set_title("variance of the mixture\n, min={}, max={}"
                                                    .format(round(avg_variance.min()),
                                                            round(avg_variance.max())))

        # reso 256
        axis[n1][0].imshow(image_1_256, vmin=0, vmax=1.0)
        axis[n1][0].set_title("source image 256")
        axis[n1][1].imshow(image_2_256, vmin=0, vmax=1.0)
        axis[n1][1].set_title("target image 256")
        axis[n1][2].imshow(remapped_est_256, vmin=0, vmax=1.0)
        axis[n1][2].set_title("source remapped with network 256")

        axis[n1][3].imshow(image_matches_gt_256.astype(np.float32), vmin=0, vmax=1)
        axis[n1][3].set_title("image matches gt 256, nbr_matches={}".format(kp2_256.shape[0]))
        axis[n1][4].imshow(image_matches_estimated_256.astype(np.float32), vmin=0, vmax=1)
        axis[n1][4].set_title("image matches estimated 256")

        if isinstance(proba_map_256, list):
            for ind in range(len(proba_map_256)):
                for ind_mode in range(num_mixture_mode):
                    proba_map_numpy = proba_map_256[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[n1 + 1 + ind][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                    axis[n1 + 1 + ind][ind_mode * 2].set_title(
                        "uncertainty_{} \n Mixture component {} Proba map \n min={}, max={}".format(ind, ind_mode,
                                                                                           round(proba_map_numpy.min()),
                                                                                           round(
                                                                                               proba_map_numpy.max())))
                    log_var_numpy = log_var_256[ind][0, ind_mode].cpu().detach().numpy().astype(np.float32)
                    axis[n1 + 1 + ind][ind_mode * 2 + 1].imshow(log_var_numpy)
                    axis[n1 + 1 + ind][ind_mode * 2 + 1].set_title(
                        "uncertainty_{} \n Mixture component {} Log Variance:\n var min={}, max={}"
                            .format(ind, ind_mode, round(np.exp(log_var_numpy).min()),
                                    round(np.exp(log_var_numpy).max())))

                avg_variance = \
                torch.sum(proba_map_256[ind].detach() * torch.exp(log_var_256[ind].detach()), dim=1, keepdim=True).squeeze()[
                    0].cpu().numpy().astype(np.float32)
                axis[n1 + 1 + ind][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
                axis[n1 + 1 + ind][num_mixture_mode * 2].set_title(
                    "uncertainty_{} \n variance of the mixture\n min={}, max={}"
                    .format(ind, round(avg_variance.min()),
                            round(avg_variance.max())))

        else:
            for ind_mode in range(num_mixture_mode):
                proba_map_numpy = proba_map_256[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[n1 + 1][ind_mode * 2].imshow(proba_map_numpy, vmin=0.0, vmax=1.0)
                axis[n1 + 1][ind_mode * 2].set_title("Mixture component {} Proba map:\n min={}, max={}".format(ind_mode,
                                                                                                       round(
                                                                                                           proba_map_numpy.min()),
                                                                                                       round(
                                                                                                           proba_map_numpy.max())))
                log_var_numpy = log_var_256[0, ind_mode].cpu().detach().numpy().astype(np.float32)
                axis[n1 + 1][ind_mode * 2 + 1].imshow(log_var_numpy)
                axis[n1 + 1][ind_mode * 2 + 1].set_title("Mixture component {} Log Variance:\n var min={}, max={}"
                                                         .format(ind_mode, round(np.exp(log_var_numpy).min()),
                                                                 round(np.exp(log_var_numpy).max())))

            avg_variance = torch.sum(proba_map_256 * torch.exp(log_var_256), dim=1, keepdim=True).squeeze()[
                0].cpu().numpy().astype(np.float32)
            axis[n1 + 1][num_mixture_mode * 2].imshow(avg_variance, vmin=1.0, vmax=10.0)
            axis[n1 + 1][num_mixture_mode * 2].set_title("variance of the mixture\n, min={}, max={}"
                                                         .format(round(avg_variance.min()),
                                                                 round(avg_variance.max())))

    fig.tight_layout()
    fig.savefig('{}/epoch{}_batch{}.png'.format(save_path, epoch, batch))
    plt.close(fig)
    return True
