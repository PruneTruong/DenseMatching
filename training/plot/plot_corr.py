import torch
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy import unravel_index
import cv2


def plot_correlation(target_image, source_image, flow_gt, correlation_volume, save_path, name,
                     exchange_source_dimensions=False, target_pts=None,
                     normalization='relu_l2norm', mask=None, image_normalization=True, plot_ind=True):
    """
    Args:
        target_image: 3xHxW
        source_image: 3xHxW
        correlation: (HxW)xH_c x W_c

    """
    # choose 10 random points
    _, H_ori, W_ori = flow_gt.shape

    # correlation can have N multiple heads, in that case, it is N, H*W, H, W
    if len(correlation_volume.shape) != 4:
        correlation_volume = correlation_volume.unsqueeze(0)  # 1, H*W, H, W
    N, _, H, W = correlation_volume.shape
    correlation_volume_original = correlation_volume.clone().detach()

    plot_occ = False
    if correlation_volume_original.shape[1] == H*W + 1:
        plot_occ = True
        occ_mask_tp_to_t = correlation_volume_original[0].permute(1, 2, 0)[:, :, -1].cpu().numpy()

    correlation_volume_original = correlation_volume_original[:, :H*W]
    if exchange_source_dimensions:
        # the correlation is source dimension first !
        correlation_volume_original = correlation_volume_original.view(N, W, H, H, W).transpose(1, 2)\
            .contiguous().view(N, H * W, H, W)

    if target_pts is None:
        nbr_pts_per_row = 2
        X, Y = np.meshgrid(np.arange(W // 5, W - 1, W // nbr_pts_per_row),
                           np.arange(H // 5, H - 1, H // nbr_pts_per_row))
        X = np.int32(X.flatten())
        Y = np.int32(Y.flatten())
    else:
        target_pts_ = target_pts.cpu().numpy().copy()
        target_pts = target_pts.cpu().numpy().copy()
        target_pts[:, 0] *= W/ W_ori
        target_pts[:, 1] *= H/H_ori
        X = target_pts[:4, 0]
        Y = target_pts[:4, 1]
        X = np.int32(X.flatten()).tolist()
        Y = np.int32(Y.flatten()).tolist()
        X.append(W//3)
        Y.append(W//3)

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)

    if plot_ind:
        correlation_volume_ = correlation_volume_original[0]
        if len(target_pts) > 1:
            pt = [X[len(X)//2], Y[len(X)//2]]
        else:
            pt = [X[0], Y[0]]
        correlation_at_point = correlation_volume_.permute(1, 2, 0).view(H, W, H, W)[pt[1], pt[0]]
        fig, axis = plt.subplots(1, 1, figsize=(20, 20))
        axis.axis('off')
        axis.imshow(correlation_at_point.cpu().numpy())
        fig.tight_layout()
        fig.savefig('{}/{}_ex_.png'.format(save_path, name),
                    bbox_inches='tight')
        plt.close(fig)

        fig, axis = plt.subplots(1, 1, figsize=(20, 20))
        axis.axis('off')
        axis.imshow(occ_mask_tp_to_t)
        fig.tight_layout()
        fig.savefig('{}/{}_ex_occ.png'.format(save_path, name),
                    bbox_inches='tight')
        plt.close(fig)

        correlation_at_point = cv2.imread('{}/{}_ex_.png'.format(save_path, name), 1)[:, :, ::-1]
        correlation_at_point = cv2.resize(correlation_at_point, source_image.shape[-2:])
        fig, axis = plt.subplots(1, 4, figsize=(20, 20))
        axis[0].imshow(target_image.squeeze().permute(1, 2, 0).numpy())
        if len(target_pts) > 1:
            axis[0].scatter(target_pts_.copy()[len(X)//2, 0],
                            target_pts_.copy()[len(X)//2, 1], s=600, color='red')
        else:
            axis[0].scatter(target_pts_.copy()[0, 0],
                            target_pts_.copy()[0, 1], s=600, color='red')
        axis[1].imshow(source_image.squeeze().permute(1, 2, 0).numpy())

        alpha = 0.1

        correlation_at_point_ = (source_image.squeeze().permute(1, 2, 0).numpy()).astype(np.uint8) * alpha + (1 - alpha) * correlation_at_point.astype(np.uint8)
        axis[2].imshow(correlation_at_point_.astype(np.uint8))

        occ_mask_tp_to_t_ = cv2.imread('{}/{}_ex_occ.png'.format(save_path, name), 1)[:, :, ::-1]
        occ_mask_tp_to_t_ = cv2.resize(occ_mask_tp_to_t_, source_image.shape[-2:])
        occ_mask_tp_to_t_ = (target_image.squeeze().permute(1, 2, 0).numpy()).astype(np.uint8) * alpha + (1 - alpha) * occ_mask_tp_to_t_.astype(np.uint8)
        axis[3].imshow(occ_mask_tp_to_t_.astype(np.uint8))
        axis[0].axis('off')
        axis[1].axis('off')
        axis[2].axis('off')
        axis[3].axis('off')
        fig.tight_layout()
        fig.savefig('{}/{}_ex.png'.format(save_path, name),
                    bbox_inches='tight')
        plt.close(fig)

    if isinstance(source_image, torch.Tensor):
        if image_normalization:
            # resizing of source and target image to correlation size
            source_image = F.interpolate(source_image.unsqueeze(0).cpu() * std_values +
                       mean_values, (H, W), mode='area').squeeze().permute(1, 2, 0).numpy()
            target_image = F.interpolate(target_image.unsqueeze(0).cpu() * std_values +
                       mean_values, (H, W), mode='area').squeeze().permute(1, 2, 0).numpy()
        else:
            # resizing of source and target image to correlation size
            source_image = F.interpolate(source_image.unsqueeze(0).cpu().float(), (H, W),
                                         mode='area').squeeze().permute(1, 2, 0).numpy()
            target_image = F.interpolate(target_image.unsqueeze(0).cpu().float(), (H, W),
                                         mode='area').squeeze().permute(1, 2, 0).numpy()
            source_image /= 255.0
            target_image /= 255.0

    flow_gt_resized = F.interpolate(flow_gt.unsqueeze(0), (H, W), mode='bilinear', align_corners=False)\
        .squeeze(0).permute(1,2,0).cpu().numpy()
    flow_gt_resized[:,:,0] *= float(W)/float(W_ori)
    flow_gt_resized[:,:,1] *= float(H)/float(H_ori)

    N_ = N + 1 if N > 1 else N

    num = 2 + N_
    if plot_occ:
        num += 1
    if mask is not None:
        fig, axis = plt.subplots(len(X)+1, num, figsize=(20, 20))
        axis[len(X)][0].imshow(mask[0].squeeze().cpu().numpy(), vmin=0, vmax=1.0)

    else:
        fig, axis = plt.subplots(len(X), num, figsize=(20, 20))


    if plot_occ:
        axis[0][num-1].imshow(occ_mask_tp_to_t, vmin=0, vmax=1.0)
        axis[0][num-1].set_title('Occlusion mask, \n min={:.2f}, max={:.2f}'.format(occ_mask_tp_to_t.min(),
                                                                                     occ_mask_tp_to_t.max()))
        axis[0][num-1].axis('off')
    for i in range(len(X)):
        pt = [X[i], Y[i]]

        # first coordinate is horizontal, second is vertical
        source_point = [int(round(pt[0] + flow_gt_resized[int(pt[1]), int(pt[0]), 0])),
                        int(round(pt[1] + flow_gt_resized[int(pt[1]), int(pt[0]), 1]))]

        axis[i][0].imshow(np.clip(target_image, 0, 1), vmin=0, vmax=1.0)
        axis[i][0].scatter(pt[0], pt[1], s=7, color='red')
        axis[i][0].set_title('target')
        axis[i][2].axis('off')

        axis[i][1].imshow(np.clip(source_image, 0, 1), vmin=0, vmax=1.0)
        if source_point[0] > W or source_point[0] < 0 or source_point[1] > H or source_point[1] < 0:
            axis[i][1].set_title('source, \npoint is outside')
        else:
            # axis[i][1].scatter(source_point[0], source_point[1], s=5, color='red')
            axis[i][1].set_title('source')

        for j in range(N_):
            if j == N and N != 1:
                correlation_volume = correlation_volume_original.mean(axis=0)
            else:
                correlation_volume = correlation_volume_original[j]
            correlation_at_point = correlation_volume.permute(1, 2, 0).view(H, W, H, W)[pt[1], pt[0]].cpu().numpy()
            max_pt_ = unravel_index(correlation_at_point.argmax(), correlation_at_point.shape)
            max_pt = [max_pt_[1], max_pt_[0]]

            min_pt_ = unravel_index(correlation_at_point.argmin(), correlation_at_point.shape)
            min_pt = [min_pt_[1], min_pt_[0]]

            min_value = np.amin(correlation_at_point)
            max_value = np.amax(correlation_at_point)
            sum = np.sum(correlation_at_point)
            if normalization == 'softmax' or normalization == 'relu_l2norm':
                axis_min_value = 0.0
                axis_max_value = 1.0
            else:
                axis_min_value = correlation_at_point.min()
                axis_max_value = correlation_at_point.max()
            axis[i][2+j].imshow(correlation_at_point, vmin=axis_min_value, vmax=axis_max_value)
            axis[i][2+j].scatter(max_pt[0],max_pt[1], s=5, color='red')
            axis[i][2+j].scatter(min_pt[0],min_pt[1], s=5, color='green')
            axis[i][2+j].set_title('max: {:.2f} (red)\n, min: {:.2f} (green)\n sum: {:.2f} '.format(
                max_value, min_value, sum ))

            if N == 1:
                axis[i][1].scatter(max_pt[0],max_pt[1], s=7, color='blue')

    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(save_path, name),
                bbox_inches='tight')
    plt.close(fig)


def plot_local_correlation(target_image, source_image, flow_gt, correlation_volume, save_path, name,
                           normalization='relu_l2norm', s=9, mask=None):
    '''

    :param target_image: 3xHxW
    :param source_image: 3xHxW
    :param correlation: (HxW)xH_c x W_c
    :return:
    '''
    # choose 10 random points
    _, H_ori, W_ori = flow_gt.shape
    _, H, W = correlation_volume.shape
    correlation_volume = correlation_volume.clone().detach()

    nbr_pts_per_row = 2
    X, Y = np.meshgrid(np.arange(W_ori // 5, W - 1, W_ori // nbr_pts_per_row),
                       np.arange(H_ori // 5, H - 1, H_ori // nbr_pts_per_row))
    X = np.round(X.flatten()).astype(np.int64)
    Y = np.round(Y.flatten()).astype(np.int64)

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)

    # resizing of source and target image to correlation size
    image_source = F.interpolate(source_image.unsqueeze(0).cpu() * std_values +
               mean_values, (H, W), mode='area').squeeze().permute(1, 2, 0).numpy()
    image_target = F.interpolate(target_image.unsqueeze(0).cpu() * std_values +
               mean_values, (H, W), mode='area').squeeze().permute(1, 2, 0).numpy()
    flow_gt_resized = F.interpolate(flow_gt.unsqueeze(0), (H, W), mode='bilinear',
                                    align_corners=False).squeeze(0).permute(1,2,0).cpu().numpy()
    flow_gt_resized[:,:,0] *= float(W)/float(W_ori)
    flow_gt_resized[:,:,1] *= float(H)/float(H_ori)

    if mask is not None:
        fig, axis = plt.subplots(len(X)+1, 3, figsize=(20, 20))
        axis[len(X)][0].imshow(mask[0].squeeze().cpu().numpy(), vmin=0, vmax=1.0)
    else:
        fig, axis = plt.subplots(len(X), 3, figsize=(20, 20))
    for i in range(len(X)):
        pt = [X[i], Y[i]]
        # first coordinate is horizontal, second is vertical
        source_point = [int(round(pt[0] + flow_gt_resized[int(pt[1]), int(pt[0]), 0])),
                        int(round(pt[1] + flow_gt_resized[int(pt[1]), int(pt[0]), 1]))]
        correlation_at_point = correlation_volume.permute(1, 2, 0).view(H, W, s, s)[pt[1], pt[0]].cpu().numpy()
        max_pt_ = unravel_index(correlation_at_point.argmax(), correlation_at_point.shape)
        max_pt = [max_pt_[1], max_pt_[0]]

        min_pt_ = unravel_index(correlation_at_point.argmin(), correlation_at_point.shape)
        min_pt = [min_pt_[1], min_pt_[0]]

        axis[i][0].imshow(np.clip(image_target, 0, 1), vmin=0, vmax=1.0)
        axis[i][0].scatter(pt[0], pt[1], s=4, color='red')
        axis[i][0].set_title('target_image, pt =({}, {})'.format(pt[0], pt[1]))

        axis[i][1].imshow(np.clip(image_source, 0, 1), vmin=0, vmax=1.0)
        if source_point[0] > W or source_point[0] < 0 or source_point[1] > H or source_point[1] < 0:
            axis[i][1].set_title('source_image, point is outside')
        else:
            axis[i][1].scatter(source_point[0], source_point[1], s=4, color='red')
            axis[i][1].set_title('source_image, pt =({}, {})'.format(source_point[0], source_point[1]))

        min_value = np.amin(correlation_at_point)
        max_value = np.amax(correlation_at_point)
        if normalization == 'softmax' or normalization == 'relu_l2norm':
            axis_min_value = 0.0
            axis_max_value = 1.0
        else:
            axis_min_value = -1.0
            axis_max_value = 1.0
        im1 = axis[i][2].imshow(correlation_at_point, vmin=axis_min_value, vmax=axis_max_value)
        axis[i][2].scatter(max_pt[0], max_pt[1], s=4, color='red')
        axis[i][2].scatter(min_pt[0], min_pt[1], s=4, color='green')
        axis[i][2].set_title('Local correlation, max is {} (red), min is {} (green)'.format(
            max_value, min_value ))
        fig.colorbar(im1, ax=axis[i][2])
    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(save_path, name),
                bbox_inches='tight')
    plt.close(fig)
    # print('saved at {}/{}.png'.format(save_path, name))