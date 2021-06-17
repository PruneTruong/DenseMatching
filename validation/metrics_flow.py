import torch


# those are for flows BxHxWx2
def compute_epe(input_flow, target_flow, mean=True, calculate_std=False):
    """
    End-point-Error computation
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
    Output:
        Averaged end-point-error (value)
    """
    EPE = torch.norm(target_flow - input_flow, p=2, dim=1)
    if calculate_std:
        EPE_std = torch.std(EPE).item()

    if mean:
        EPE = EPE.mean().item()
        # shape is BxHxWx2

    if calculate_std:
        return EPE, EPE_std
    else:
        return EPE


def correct_correspondences(input_flow, target_flow, alpha, img_size, epe_tensor=None):
    """
    Computation PCK, i.e number of the pixels within a certain threshold
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    if epe_tensor is not None:
        dist = epe_tensor
    else:
        dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    pck_threshold = alpha * img_size
    mask = dist.le(pck_threshold) # Computes dist ≤ pck_threshold element-wise (element then equal to 1)
    return mask.sum().item()


def F1_kitti_2015(input_flow, target_flow, tau=[3.0, 0.05]):
    """
    Computation number of outliers
    for which error > 3px(tau[0]) and error/magnitude(ground truth flow) > 0.05(tau[1])
    Args:
        input_flow: estimated flow [BxHxW,2]
        target_flow: ground-truth flow [BxHxW,2]
        alpha: threshold
        img_size: image size
    Output:
        PCK metric
    """
    # input flow is shape (BxHgtxWgt,2)
    dist = torch.norm(target_flow - input_flow, p=2, dim=1)
    gt_magnitude = torch.norm(target_flow, p=2, dim=1)
    # dist is shape BxHgtxWgt
    mask = dist.gt(3.0) & (dist/gt_magnitude).gt(0.05) # Computes dist > 3 and
    return mask.sum().item()


def calculate_speed(flow, mask_valid):
    speed = torch.sqrt(flow[:,0,:,:]**2 + flow[:,1,:,:]**2)
    mask_small = speed.lt(10) & mask_valid
    mask_medium = speed.ge(10) & speed.lt(40) & mask_valid
    mask_large = speed.ge(40) & mask_valid
    return mask_small, mask_medium, mask_large
