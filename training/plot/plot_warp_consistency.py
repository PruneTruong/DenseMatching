import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from utils_flow.pixel_wise_mapping import remap_using_flow_fields
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.flow_and_mapping_operations import convert_flow_to_mapping


def plot_flows_warpc(save_path, name, h, w, image_source, image_target, image_target_prime,
                     estimated_flow_target_to_source, estimated_flow_target_prime_to_source,
                     estimated_flow_target_prime_to_target, estimated_flow_target_prime_to_target_directly,
                     gt_flow_target_prime_to_target, gt_flow_target_to_source=None,
                     estimated_flow_source_to_target=None, sparse=False, mask=None, image_target_prime_ss=None,
                     mask_cyclic=None):

    max_flow = 150
    max_mapping = 257

    # unnormalize the images
    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=image_source.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=image_source.dtype).view(3, 1, 1)

    image_source = (image_source[0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    image_target = (image_target[0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    image_target_prime = (image_target_prime[0].cpu() * std_values + mean_values).clamp(0, 1).permute(1, 2, 0).numpy()
    if image_target_prime_ss is not None:
        image_target_prime_ss = (image_target_prime_ss[0].cpu() * std_values + mean_values).clamp(0, 1)\
            .permute(1, 2, 0).numpy()

    if estimated_flow_target_to_source is None:
        estimated_flow_target_prime_to_target_directly = F.interpolate(estimated_flow_target_prime_to_target_directly,
                                                                       (h, w), mode='bilinear', align_corners=False)
        # shape Bx2xHxW
        estimated_flow_target_prime_to_target_directly = \
            estimated_flow_target_prime_to_target_directly.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
        gt_flow_target_prime_to_target = gt_flow_target_prime_to_target.detach().permute(0, 2, 3, 1)[0].cpu().numpy()

        image_target_to_target_prime_est_directly = \
            remap_using_flow_fields(image_target, estimated_flow_target_prime_to_target_directly[:, :, 0],
                                    estimated_flow_target_prime_to_target_directly[:, :, 1])
        image_target_to_target_prime_gt = remap_using_flow_fields(image_target, gt_flow_target_prime_to_target[:, :, 0],
                                                                  gt_flow_target_prime_to_target[:, :, 1])

        fig, axis = plt.subplots(1, 5, figsize=(20, 20))
        axis[0].imshow(image_target)
        axis[0].set_title("target image")
        axis[1].imshow(image_target_prime)
        axis[1].set_title("target image prime")
        axis[2].imshow(image_target_to_target_prime_est_directly)
        axis[2].set_title('estimated directly\n target remapped to target prime')
        axis[3].imshow(image_target_to_target_prime_gt)
        axis[3].set_title('gt\n target remapped to target prime')
        if mask is not None:
            mask = mask.cpu().numpy()[0].astype(np.float32)
            axis[4].imshow(mask, vmin=0, vmax=1)
            axis[4].set_title('mask applied during training')

        fig.savefig('{}/{}.png'.format(save_path, name),
                    bbox_inches='tight')
        plt.close(fig)
    else:
        estimated_flow_target_to_source = F.interpolate(estimated_flow_target_to_source, (h, w), mode='bilinear',
                                                        align_corners=False)  # shape Bx2xHxW
        if estimated_flow_target_prime_to_source is not None:
            estimated_flow_target_prime_to_source = F.interpolate(estimated_flow_target_prime_to_source, (h, w),
                                                                  mode='bilinear', align_corners=False)  # shape Bx2xHxW
        estimated_flow_target_prime_to_target = F.interpolate(estimated_flow_target_prime_to_target, (h, w),
                                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW
        estimated_flow_target_prime_to_target_directly = F.interpolate(estimated_flow_target_prime_to_target_directly, (h, w),
                                                              mode='bilinear', align_corners=False)  # shape Bx2xHxW

        estimated_flow_target_to_source = estimated_flow_target_to_source.detach().permute(0, 2, 3, 1)[0].cpu().numpy()
        if estimated_flow_target_prime_to_source is not None:
            estimated_flow_target_prime_to_source = estimated_flow_target_prime_to_source.detach()\
                .permute(0, 2, 3, 1)[0].cpu().numpy()
        estimated_flow_target_prime_to_target = estimated_flow_target_prime_to_target.detach()\
            .permute(0, 2, 3, 1)[0].cpu().numpy()
        estimated_flow_target_prime_to_target_directly = estimated_flow_target_prime_to_target_directly\
            .detach().permute(0, 2, 3, 1)[0].cpu().numpy()
        gt_flow_target_prime_to_target = gt_flow_target_prime_to_target.detach().permute(0, 2, 3, 1)[0].cpu().numpy()

        estimated_flow_target_to_source_rgb = flow_to_image(estimated_flow_target_to_source, max_flow)
        if estimated_flow_target_prime_to_source is not None:
            estimated_flow_target_prime_to_source_rgb = flow_to_image(estimated_flow_target_prime_to_source, max_flow)
        estimated_flow_target_prime_to_target_rgb = flow_to_image(estimated_flow_target_prime_to_target, max_flow)
        estimated_flow_target_prime_to_target_directly_rgb = \
            flow_to_image(estimated_flow_target_prime_to_target_directly, max_flow)
        gt_flow_target_prime_to_target_rgb = flow_to_image(gt_flow_target_prime_to_target, max_flow)

        estimated_mapping_target_to_source_rgb = flow_to_image(convert_flow_to_mapping(estimated_flow_target_to_source,
                                                                                       False), max_mapping)
        if estimated_flow_target_prime_to_source is not None:
            estimated_mapping_target_prime_to_source_rgb = \
                flow_to_image(convert_flow_to_mapping(estimated_flow_target_prime_to_source, False), max_mapping)
        estimated_mapping_target_prime_to_target_rgb = \
            flow_to_image(convert_flow_to_mapping(estimated_flow_target_prime_to_target, False), max_mapping)
        estimated_mapping_target_prime_to_target_directly_rgb = \
            flow_to_image(convert_flow_to_mapping(estimated_flow_target_prime_to_target_directly, False), max_mapping)
        gt_mapping_target_prime_to_target_rgb = \
            flow_to_image(convert_flow_to_mapping(gt_flow_target_prime_to_target, False), max_mapping)

        image_source_to_target_est = remap_using_flow_fields(image_source, estimated_flow_target_to_source[:,:,0],
                                                             estimated_flow_target_to_source[:, :, 1])
        if estimated_flow_target_prime_to_source is not None:
            image_source_to_target_prime_est = \
                remap_using_flow_fields(image_source, estimated_flow_target_prime_to_source[:, :, 0],
                                        estimated_flow_target_prime_to_source[:, :, 1])
        image_target_to_target_prime_est = \
            remap_using_flow_fields(image_target, estimated_flow_target_prime_to_target[:, :, 0],
                                    estimated_flow_target_prime_to_target[:, :, 1])
        image_target_to_target_prime_est_directly = \
            remap_using_flow_fields(image_target, estimated_flow_target_prime_to_target_directly[:, :, 0],
                                    estimated_flow_target_prime_to_target_directly[:, :, 1])
        image_target_to_target_prime_gt = remap_using_flow_fields(image_target, gt_flow_target_prime_to_target[:, :, 0],
                                                                  gt_flow_target_prime_to_target[:, :, 1])

        if estimated_flow_source_to_target is not None:
            fig, axis = plt.subplots(6, 5, figsize=(20, 20))
            estimated_flow_source_to_target = F.interpolate(estimated_flow_source_to_target, (h, w), mode='bilinear',
                                                            align_corners=False)
            estimated_flow_source_to_target = estimated_flow_source_to_target.detach().permute(0, 2, 3, 1)[0].cpu().numpy()

            estimated_flow_source_to_target_rgb = flow_to_image(estimated_flow_source_to_target, max_flow)
            estimated_mapping_source_to_target_rgb = flow_to_image(convert_flow_to_mapping(estimated_flow_source_to_target, False), max_mapping)
            image_target_to_source_est = remap_using_flow_fields(image_target, estimated_flow_source_to_target[:, :, 0],
                                                                 estimated_flow_source_to_target[:, :, 1])
            axis[2][0].imshow(image_target)
            axis[2][0].set_title("target image")
            axis[2][1].imshow(image_source)
            axis[2][1].set_title("source image")
            axis[2][2].imshow(image_target_to_source_est)
            axis[2][2].set_title('target remapped to source')
            axis[2][3].imshow(estimated_flow_source_to_target_rgb)
            axis[2][3].set_title('flow est from source to target')
            axis[2][4].imshow(estimated_mapping_source_to_target_rgb)
            axis[2][4].set_title('mapping est from source to target')
            n = 2
        else:
            fig, axis = plt.subplots(5, 5, figsize=(20, 20))
            n = 1

        axis[0][0].imshow(image_source)
        axis[0][0].set_title("source image")
        axis[0][1].imshow(image_target)
        axis[0][1].set_title("target image")
        axis[0][2].imshow(image_source_to_target_est)
        axis[0][2].set_title('source remapped to target')
        axis[0][3].imshow(estimated_flow_target_to_source_rgb)
        axis[0][3].set_title('flow est from target to source ')
        axis[0][4].imshow(estimated_mapping_target_to_source_rgb)
        axis[0][4].set_title('mapping est from target to source ')

        if estimated_flow_target_prime_to_source is not None:
            axis[1][0].imshow(image_source)
            axis[1][0].set_title("source image")
            axis[1][1].imshow(image_target_prime)
            axis[1][1].set_title("target image prime")
            axis[1][2].imshow(image_source_to_target_prime_est)
            axis[1][2].set_title('source remapped to target prime')
            axis[1][3].imshow(estimated_flow_target_prime_to_source_rgb)
            axis[1][3].set_title('flow est from target prime to source ')
            axis[1][4].imshow(estimated_mapping_target_prime_to_source_rgb)
            axis[1][4].set_title('mapping est from target primeto source')

        axis[n+1][0].imshow(image_target)
        axis[n+1][0].set_title("target image")
        axis[n+1][1].imshow(image_target_prime)
        axis[n+1][1].set_title("target image prime")
        axis[n+1][2].imshow(image_target_to_target_prime_est)
        axis[n+1][2].set_title('composition:\n target remapped to target prime')
        axis[n+1][3].imshow(estimated_flow_target_prime_to_target_rgb)
        axis[n+1][3].set_title('composition:\n flow est from target prime to target')
        axis[n+1][4].imshow(estimated_mapping_target_prime_to_target_rgb)
        axis[n+1][4].set_title('composition:\n mapping est from target prime to target')

        if mask_cyclic is not None:
            mask_cyclic = mask_cyclic.cpu().numpy()[0].astype(np.float32)
            axis[n+2][0].imshow(mask_cyclic, vmin=0, vmax=1)
            axis[n+2][0].set_title('mask from cyclic consistency')

        if mask is not None:
            mask = mask.cpu().numpy()[0].astype(np.float32)
            axis[n+2][1].imshow(mask, vmin=0, vmax=1)
            axis[n+2][1].set_title('mask applied during training')
        axis[n+2][2].imshow(image_target_to_target_prime_gt)
        axis[n+2][2].set_title('GROUND-truth:\n target remapped to target prime')
        axis[n+2][3].imshow(gt_flow_target_prime_to_target_rgb)
        axis[n+2][3].set_title('GROUND-truth:\n flow from target prime to target')
        axis[n+2][4].imshow(gt_mapping_target_prime_to_target_rgb)
        axis[n+2][4].set_title('GROUND-truth:\n mapping from target prime to target')

        # direct estimation network
        axis[n+3][0].imshow(image_target)
        axis[n+3][0].set_title("target image")
        if image_target_prime_ss is not None:
            axis[n+3][1].imshow(image_target_prime_ss)
        else:
            axis[n+3][1].imshow(image_target_prime)
        axis[n+3][1].set_title("target image prime")
        axis[n+3][2].imshow(image_target_to_target_prime_est_directly)
        axis[n+3][2].set_title('estimated directly\n target remapped to target prime')
        axis[n+3][3].imshow(estimated_flow_target_prime_to_target_directly_rgb)
        axis[n+3][3].set_title('estimated directly\n flow est from target prime to target')
        axis[n+3][4].imshow(estimated_mapping_target_prime_to_target_directly_rgb)
        axis[n+3][4].set_title('estimated directly\n mapping est from target prime to target')

        fig.tight_layout()
        fig.savefig('{}/{}.png'.format(save_path, name))
        plt.close(fig)
