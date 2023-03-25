import torch
import torch.nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
from packaging import version

from models.modules.feature_correlation_layer import compute_global_correlation, featureL2Norm
from utils_flow.flow_and_mapping_operations import (convert_mapping_to_flow, convert_flow_to_mapping,
                                                    unormalise_and_convert_mapping_to_flow)


def normalize_image_with_imagenet_weights(source_img):
    # img has shape bx3xhxw
    b, _, h_scale, w_scale = source_img.shape
    mean_vector = [0.485, 0.456, 0.406]
    std_vector = [0.229, 0.224, 0.225]

    # original resolution
    source_img_copy = source_img.float().div(255.0)
    mean = torch.as_tensor(mean_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    std = torch.as_tensor(std_vector, dtype=source_img_copy.dtype, device=source_img_copy.device)
    source_img_copy.sub_(mean[:, None, None]).div_(std[:, None, None])
    return source_img_copy


def apply_gaussian_kernel(corr, sigma=5):
    h, w = corr.shape[-2:]
    b = corr.shape[0]
    corr = corr.view(b, -1, h, w)
    hw = corr.shape[1]

    idx = corr.max(dim=1)[1]  # b x h x w    get maximum value along channel
    idx_y = (idx // w).view(b, 1, 1, h, w).float()
    idx_x = (idx % w).view(b, 1, 1, h, w).float()

    x = np.linspace(0, w - 1, w)
    x = torch.tensor(x, dtype=torch.float, requires_grad=False).to(corr.device)
    x = x.view(1, 1, w, 1, 1).expand(b, 1, w, h, w)

    y = np.linspace(0, h - 1, h)
    y = torch.tensor(y, dtype=torch.float, requires_grad=False).to(corr.device)
    y = y.view(1, h, 1, 1, 1).expand(b, h, 1, h, w)

    gauss_kernel = torch.exp(-((x - idx_x) ** 2 + (y - idx_y) ** 2) / (2 * sigma ** 2))
    gauss_kernel = gauss_kernel.view(b, hw, h, w)

    return gauss_kernel * corr


def correlation_to_flow_w_soft_argmax(correlation_target_to_source, output_shape, temperature=1.0, apply_softmax=True,
                                      stable_softmax=False):
    """
    Convert correlation to flow, with soft argmax.
    Modified from SFNet: Learning Object-aware Semantic Flow (Lee et al.)
    Args:
        correlation_target_to_source: shape is B, H_s*W_s, H_t, W_t
        output_shape: output shape of the flow relating the target to the source image (H, W)
        temperature: to apply in the softmax operation
        apply_softmax: bool. Otherwise, the softmax was applied before.
        stable_softmax: bool. use stable softmax?

    Returns:
        flow_est: flow field relating the target to the source, at output_shape.
    """

    def softmax_with_temperature(x, temperature_, d=1):
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        return F.softmax(x / temperature_, dim=1)

    def soft_argmax(corr, temperature_, apply_softmax_, stable_softmax_):
        b, _, h, w = corr.size()

        if apply_softmax_:
            if stable_softmax_:
                corr = softmax_with_temperature(corr, temperature_=temperature_, d=1)
            else:
                corr = F.softmax(corr / temperature_, dim=1)
        else:
            # here this is supposed to be the results of softmax. sum(dim=1) equal to 1!
            # but not the case for bin. divide by sum so sum equal to 1
            corr = corr / (corr.sum(dim=1, keepdim=True) + 1e-8)
        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        x_normal = torch.linspace(-1, 1, w).to(corr.device)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        y_normal = torch.linspace(-1, 1, h).to(corr.device)
        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
        return grid_x, grid_y

    B = correlation_target_to_source.shape[0]
    if len(correlation_target_to_source.shape) == 3:
        h = w = int(math.sqrt(correlation_target_to_source.shape[-1]))
    else:
        h, w = correlation_target_to_source.shape[-2:]
    grid_x, grid_y = soft_argmax(correlation_target_to_source.view(B, -1, h, w), temperature_=temperature,
                                 apply_softmax_=apply_softmax,
                                 stable_softmax_=stable_softmax)

    flow = torch.cat((grid_x, grid_y), dim=1)
    flow = unormalise_and_convert_mapping_to_flow(flow)  # at the resolution of the correlation

    flow_est = F.interpolate(flow, size=output_shape, mode='bilinear', align_corners=False)
    flow_est[:, 0] *= output_shape[1] / w
    flow_est[:, 1] *= output_shape[0] / h
    return flow_est


def correlation_to_flow_w_argmax(correlation_target_to_source, output_shape=None, return_mapping=False, do_softmax=False):
    """
    Convert correlation to flow, with argmax.
    Args:
        correlation_target_to_source: shape is B, H_s*W_s, H_t, W_t
        output_shape: output shape of the flow from the target to the source image (H, W)
        do_softmax: bool, apply softmax to the correlation before finding the best match? (should not change anything)
        return_mapping: bool

    Returns:
        if return_mapping:
            correspondence map relating the target to the source, at output_shape.
        else:
            flow_est: flow field relating the target to the source, at output_shape.
    """
    H, W = correlation_target_to_source.shape[-2:]
    b = correlation_target_to_source.shape[0]
    # get matches corresponding to maximum in correlation
    (x_source, y_source, x_target, y_target, score) = corr_to_matches(
        correlation_target_to_source.view(b, H, W, H, W).unsqueeze(1),
        get_maximum=True, do_softmax=do_softmax)

    # x_source dimension is B x H*W
    mapping_est = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1).view(b, H, W, 2).permute(0, 3, 1, 2)
    # score = score.view(b, H, W)

    # b, 2, H, W
    flow_est = convert_mapping_to_flow(mapping_est)

    if output_shape is not None and (H != output_shape[0] or W != output_shape[1]):
        flow_est = F.interpolate(flow_est, output_shape, mode='bilinear', align_corners=False)
        flow_est[:, 0] *= float(output_shape[1]) / float(W)
        flow_est[:, 1] *= float(output_shape[0]) / float(H)
    if return_mapping:
        return convert_flow_to_mapping(flow_est)
    else:
        return flow_est


def estimate_epe_from_correlation(correlation, flow_gt, mask_gt, pck_thresh_1=1.0, pck_thresh_2=3.0, do_softmax=False):
    """
    Compute metrics after converting correlation to flow, with argmax.
    Args:
        correlation: shape is B, H_s*W_s, H_t, W_t
        flow_gt: shape is B, 2, H, W
        mask_gt: shape is B, H, W
        pck_thresh_1:
        pck_thresh_2:
        do_softmax: bool, apply softmax to the correlation before finding the best match? (should not change anything)

    Returns:
        aepe and pck metrics.
    """
    if len(mask_gt.shape) == 4:
        mask_gt = mask_gt.squeeze(1)

    flow_gt = flow_gt.permute(0, 2, 3, 1)
    _, h_flow, w_flow, _ = flow_gt.shape

    # correlation shape H*W, H, W
    b, _, H, W = correlation.shape

    # get matches corresponding to maximum in correlation
    (x_source, y_source, x_target, y_target, score) = corr_to_matches(correlation.view(b, H, W, H, W).unsqueeze(1),
                                                                      get_maximum=True, do_softmax=do_softmax)

    # x_source dimension is B x H*W
    mapping_est = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1).view(b, H, W, 2)
    flow_est = convert_mapping_to_flow(mapping_est)
    # b, 2, H, W

    # mapping_gt shape b, H,W,2 ==> index of the position is in those dimensions
    if h_flow != H or w_flow != W:
        flow_est = F.interpolate(flow_est, (h_flow, w_flow), mode='bilinear',
                                 align_corners=False)
        flow_est[:, 0] *= float(w_flow) / float(W)
        flow_est[:, 1] *= float(h_flow) / float(H)
    flow_est = flow_est.permute(0, 2, 3, 1)  # B, h, w, 2

    if mask_gt is not None:
        mask_gt = F.interpolate(mask_gt.unsqueeze(1).float(), (h_flow, w_flow), mode='bilinear',
                                align_corners=False).squeeze(1).byte()
        mask_gt = mask_gt.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask_gt.byte()

        flow_gt = flow_gt[mask_gt]  # N, 2
        flow_est = flow_est[mask_gt]

    epe = torch.sum((flow_gt - flow_est) ** 2, dim=-1).sqrt()
    if len(mapping_est) > 0:
        aepe = epe.mean().item()
        pck_1 = epe.le(pck_thresh_1).float().mean().item()
        pck_3 = epe.le(pck_thresh_2).float().mean().item()
    else:
        aepe, pck_1, pck_3 = 0.0, 0.0, 0.0

    return aepe, pck_1, pck_3


def features_to_flow(c_source, c_target, output_shape=None, return_mapping=False):
    """
    From two sets of feature maps, compute the flow relating them (via argmax of the global correlation).
    Correlation built after normalizing the features.
    Args:
        c_source: B, C, H_s, W_s
        c_target: B, C, H_t, W_t
        output_shape: output shape of the flow from the target to the source image (H, W)
        return_mapping: bool

    Returns:
        if return_mapping:
            correspondence map relating the target to the source, at output_shape B, 2, H, W
        else:
            flow_est: flow field relating the target to the source, at output_shape B, 2, H, W
    """
    correlation_volume = compute_global_correlation(feature_source=featureL2Norm(c_source),
                                                    feature_target=featureL2Norm(c_target))
    # b, h_s*w_s, h_t, w_t
    return correlation_to_flow_w_argmax(correlation_volume, output_shape=output_shape, return_mapping=return_mapping)


def corr_to_matches(corr4d, delta4d=None, k_size=1, do_softmax=False, scale='positive', return_indices=False,
                    invert_matching_direction=False, get_maximum=True):
    """
    Modified from NC-Net. Perform argmax over the correlation.
    Args:
        corr4d: correlation, shape is b, 1, H_s, W_s, H_t, W_t
        delta4d:
        k_size:
        do_softmax:
        scale:
        return_indices:
        invert_matching_direction:
        get_maximum:

    Returns:

    """
    to_cuda = lambda x: x.cuda() if corr4d.is_cuda else x
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    if scale == 'centered':
        XA, YA = np.meshgrid(np.linspace(-1, 1, fs2 * k_size), np.linspace(-1, 1, fs1 * k_size))
        XB, YB = np.meshgrid(np.linspace(-1, 1, fs4 * k_size), np.linspace(-1, 1, fs3 * k_size))
    elif scale == 'positive':
        # keep normal range of coordinate
        XA, YA = np.meshgrid(np.linspace(0, fs2 - 1, fs2 * k_size), np.linspace(0, fs1 - 1, fs1 * k_size))
        XB, YB = np.meshgrid(np.linspace(0, fs4 - 1, fs4 * k_size), np.linspace(0, fs3 - 1, fs3 * k_size))

    JA, IA = np.meshgrid(range(fs2), range(fs1))
    JB, IB = np.meshgrid(range(fs4), range(fs3))

    XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(to_cuda(torch.FloatTensor(YA)))
    XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(to_cuda(torch.FloatTensor(YB)))

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).view(1, -1))), Variable(to_cuda(torch.LongTensor(IA).view(1, -1)))
    JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    if invert_matching_direction:
        nc_A_Bvec = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

        if do_softmax:
            nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec, dim=3)

        if get_maximum:
            match_A_vals, idx_A_Bvec = torch.max(nc_A_Bvec, dim=3)
        else:
            match_A_vals, idx_A_Bvec = torch.min(nc_A_Bvec, dim=3)
        score = match_A_vals.view(batch_size, -1)

        iB = IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        jB = JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        iA = IA.expand_as(iB)
        jA = JA.expand_as(jB)

    else:
        nc_B_Avec = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
        if do_softmax:
            nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, dim=1)

        if get_maximum:
            match_B_vals, idx_B_Avec = torch.max(nc_B_Avec, dim=1)
        else:
            match_B_vals, idx_B_Avec = torch.min(nc_B_Avec, dim=1)
        score = match_B_vals.view(batch_size, -1)

        iA = IA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
        jA = JA.view(-1)[idx_B_Avec.view(-1)].view(batch_size, -1)
        iB = IB.expand_as(iA)
        jB = JB.expand_as(jA)

    if delta4d is not None:  # relocalization
        delta_iA, delta_jA, delta_iB, delta_jB = delta4d

        diA = delta_iA.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]
        djA = delta_jA.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]
        diB = delta_iB.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]
        djB = delta_jB.squeeze(0).squeeze(0)[iA.view(-1), jA.view(-1), iB.view(-1), jB.view(-1)]

        iA = iA * k_size + diA.expand_as(iA)
        jA = jA * k_size + djA.expand_as(jA)
        iB = iB * k_size + diB.expand_as(iB)
        jB = jB * k_size + djB.expand_as(jB)

    xA = XA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    yA = YA[iA.view(-1), jA.view(-1)].view(batch_size, -1)
    xB = XB[iB.contiguous().view(-1), jB.contiguous().view(-1)].view(batch_size, -1)
    yB = YB[iB.contiguous().view(-1), jB.contiguous().view(-1)].view(batch_size, -1)

    # XA is index in channel dimension (source)
    if return_indices:
        return xA, yA, xB, yB, score, iA, jA, iB, jB
    else:
        return xA, yA, xB, yB, score


###########################  Correlation classes ##############################

def cost_volume_to_probabilistic_mapping(A, activation, temperature):
    """ Convert cost volume to probabilistic mapping.
    Args:
        A: cost volume, dimension B x C x H x W, matching points are in C
        activation: function to convert the cost volume to a probabilistic mapping
        temperature: to apply to the cost volume scores before the softmax function
    """
    def l1normalize(x):
        r"""L1-normalization"""
        vector_sum = torch.sum(x, dim=1, keepdim=True)
        vector_sum[vector_sum == 0] = 1.0
        return x / vector_sum

    if activation == 'softmax':
        proba = F.softmax(A / temperature, dim=1)
    elif activation == 'unit_gaussian_softmax':
        A = Norm.unit_gaussian_normalize(A, dim=1)
        proba = F.softmax(A / temperature, dim=1)
    elif activation == 'stable_softmax':
        M, _ = A.max(dim=1, keepdim=True)
        A = A - M  # subtract maximum value for stability
        return F.softmax(A / temperature, dim=1)
    elif activation == 'l1norm':
        proba = l1normalize(A)
    elif activation == 'l1norm_nn_mutual':
        b, c, h, w = A.shape
        A = Correlation.mutual_nn_filter(A.view(b, c, -1)).view(b, -1, h, w)
        proba = l1normalize(A)
    elif activation == 'noactivation':
        proba = A
    else:
        raise ValueError
    return proba


class Correlation:
    @classmethod
    def bmm_interp(cls, src_feat, trg_feat, interp_size):
        r"""Performs batch-wise matrix-multiplication after interpolation"""
        src_feat = F.interpolate(src_feat, interp_size, mode='bilinear', align_corners=True)
        trg_feat = F.interpolate(trg_feat, interp_size, mode='bilinear', align_corners=True)

        src_feat = src_feat.view(src_feat.size(0), src_feat.size(1), -1).transpose(1, 2)
        trg_feat = trg_feat.view(trg_feat.size(0), trg_feat.size(1), -1)

        return torch.bmm(src_feat, trg_feat)

    @classmethod
    def mutual_nn_filter(cls, correlation_matrix):
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=1, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)


class Norm:
    r"""Vector normalization"""
    @classmethod
    def feat_normalize(cls, x, interp_size):
        r"""L2-normalizes given 2D feature map after interpolation"""
        x = F.interpolate(x, interp_size, mode='bilinear', align_corners=True)
        return x.pow(2).sum(1).view(x.size(0), -1)

    @classmethod
    def l1normalize(cls, x):
        r"""L1-normalization"""
        vector_sum = torch.sum(x, dim=2, keepdim=True)
        vector_sum[vector_sum == 0] = 1.0
        return x / vector_sum

    @classmethod
    def unit_gaussian_normalize(cls, x, dim=2):
        r"""Make each (row) distribution into unit gaussian"""
        correlation_matrix = x - x.mean(dim=dim).unsqueeze(dim).expand_as(x)

        with torch.no_grad():
            standard_deviation = correlation_matrix.std(dim=dim)
            standard_deviation[standard_deviation == 0] = 1.0
        correlation_matrix /= standard_deviation.unsqueeze(dim).expand_as(correlation_matrix)

        return correlation_matrix


##########################  from NC-Net  ######################################


def normalize_axis(x, L):
    return (x - 1 - (L - 1) / 2) * 2 / (L - 1)


def unnormalize_axis(x, L):
    return x * (L - 1) / 2 + 1 + (L - 1) / 2


def nearestNeighPointTnf(matches, target_points_norm):
    xA, yA, xB, yB = matches

    # match target points to grid
    deltaX = target_points_norm[:, 0, :].unsqueeze(1) - xB.unsqueeze(2)
    deltaY = target_points_norm[:, 1, :].unsqueeze(1) - yB.unsqueeze(2)
    distB = torch.sqrt(torch.pow(deltaX, 2) + torch.pow(deltaY, 2))
    vals, idx = torch.min(distB, dim=1)

    warped_points_x = xA.view(-1)[idx.view(-1)].view(1, 1, -1)
    warped_points_y = yA.view(-1)[idx.view(-1)].view(1, 1, -1)
    warped_points_norm = torch.cat((warped_points_x, warped_points_y), dim=1)
    return warped_points_norm


def bilinearInterpPointTnf(matches, target_points_norm):
    xA, yA, xB, yB = matches

    feature_size = int(np.sqrt(xB.shape[-1]))

    b, _, N = target_points_norm.size()

    X_ = xB.view(-1)
    Y_ = yB.view(-1)

    grid = torch.FloatTensor(np.linspace(-1, 1, feature_size)).unsqueeze(0).unsqueeze(2)
    if xB.is_cuda:
        grid = grid.cuda()
    if isinstance(xB, Variable):
        grid = Variable(grid)

    x_minus = torch.sum(((target_points_norm[:, 0, :] - grid) > 0).long(), dim=1, keepdim=True) - 1
    x_minus[x_minus < 0] = 0  # fix edge case
    x_plus = x_minus + 1

    y_minus = torch.sum(((target_points_norm[:, 1, :] - grid) > 0).long(), dim=1, keepdim=True) - 1
    y_minus[y_minus < 0] = 0  # fix edge case
    y_plus = y_minus + 1

    toidx = lambda x, y, L: y * L + x

    m_m_idx = toidx(x_minus, y_minus, feature_size)
    p_p_idx = toidx(x_plus, y_plus, feature_size)
    p_m_idx = toidx(x_plus, y_minus, feature_size)
    m_p_idx = toidx(x_minus, y_plus, feature_size)

    topoint = lambda idx, X, Y: torch.cat((X[idx.view(-1)].view(b, 1, N).contiguous(),
                                           Y[idx.view(-1)].view(b, 1, N).contiguous()), dim=1)

    P_m_m = topoint(m_m_idx, X_, Y_)
    P_p_p = topoint(p_p_idx, X_, Y_)
    P_p_m = topoint(p_m_idx, X_, Y_)
    P_m_p = topoint(m_p_idx, X_, Y_)

    multrows = lambda x: x[:, 0, :] * x[:, 1, :]

    f_p_p = multrows(torch.abs(target_points_norm - P_m_m))
    f_m_m = multrows(torch.abs(target_points_norm - P_p_p))
    f_m_p = multrows(torch.abs(target_points_norm - P_p_m))
    f_p_m = multrows(torch.abs(target_points_norm - P_m_p))

    Q_m_m = topoint(m_m_idx, xA.view(-1), yA.view(-1))
    Q_p_p = topoint(p_p_idx, xA.view(-1), yA.view(-1))
    Q_p_m = topoint(p_m_idx, xA.view(-1), yA.view(-1))
    Q_m_p = topoint(m_p_idx, xA.view(-1), yA.view(-1))

    warped_points_norm = (Q_m_m * f_m_m + Q_p_p * f_p_p + Q_m_p * f_m_p + Q_p_m * f_p_m) / (
                f_p_p + f_m_m + f_m_p + f_p_m)
    return warped_points_norm


def PointsToUnitCoords(P, im_size):
    h, w = im_size[:, 0], im_size[:, 1]
    P_norm = P.clone()
    # normalize Y
    P_norm[:, 0, :] = normalize_axis(P[:, 0, :], w.unsqueeze(1).expand_as(P[:, 0, :]))
    # normalize X
    P_norm[:, 1, :] = normalize_axis(P[:, 1, :], h.unsqueeze(1).expand_as(P[:, 1, :]))
    return P_norm


def PointsToPixelCoords(P, im_size):
    h, w = im_size[:, 0], im_size[:, 1]
    P_norm = P.clone()
    # normalize Y
    P_norm[:, 0, :] = unnormalize_axis(P[:, 0, :], w.unsqueeze(1).expand_as(P[:, 0, :]))
    # normalize X
    P_norm[:, 1, :] = unnormalize_axis(P[:, 1, :], h.unsqueeze(1).expand_as(P[:, 1, :]))
    return P_norm


def get_dense_flow_from_correlation(correlation_from_t_to_s, output_shape, device):
    """
    Alternative to convert correlation to flow, with soft argmax.
    Extracted from NC-Net.
    Args:
        correlation_from_t_to_s: shape is B, H_s*W_s, H_t, W_t
        output_shape: output shape of the flow from the target to the source image (H, W)
        device: gpu or cpu

    Returns:
        flow_est: flow field relating the target to the source, at output_shape.
    """
    xA, yA, xB, yB, sB = corr_to_matches(correlation_from_t_to_s, do_softmax=True, scale='centered')
    matches = (xA, yA, xB, yB)
    h_tgt, w_tgt = output_shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w_tgt - 1, w_tgt),
                       np.linspace(0, h_tgt - 1, h_tgt))

    grid_X_vec = torch.from_numpy(X).view(-1, 1).float()
    grid_Y_vec = torch.from_numpy(Y).view(-1, 1).float()

    grid_XY_vec = torch.cat((grid_X_vec, grid_Y_vec), 1).permute(1, 0).to(device)

    target_points_ = grid_XY_vec.to(device)
    target_points_norm = PointsToUnitCoords(target_points_.unsqueeze(0),
                                            torch.Tensor(output_shape).unsqueeze(0).to(device))
    # 1, 2, N

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches, target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm, torch.Tensor(output_shape).unsqueeze(0).to(device))
    warped_points = torch.t(warped_points.squeeze())
    # N, 2
    warped_points = warped_points.reshape(output_shape[0], output_shape[1], 2)
    return warped_points.permute(2, 0, 1) - grid_XY_vec.reshape(2, output_shape[0], output_shape[1])


def interpolation(matches, output_shape, device):
    h_tgt, w_tgt = output_shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w_tgt - 1, w_tgt),
                       np.linspace(0, h_tgt - 1, h_tgt))

    grid_X_vec = torch.from_numpy(X).view(-1, 1).float()
    grid_Y_vec = torch.from_numpy(Y).view(-1, 1).float()

    grid_XY_vec = torch.cat((grid_X_vec, grid_Y_vec), 1).permute(1, 0).to(device)

    target_points_ = grid_XY_vec.to(device)
    target_points_norm = PointsToUnitCoords(target_points_.unsqueeze(0),
                                            torch.Tensor(output_shape).unsqueeze(0).to(device))
    # 1, 2, N

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches, target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm, torch.Tensor(output_shape).unsqueeze(0).to(device))
    warped_points = torch.t(warped_points.squeeze())
    # N, 2
    warped_points = warped_points.reshape(output_shape[0], output_shape[1], 2)
    return warped_points.permute(2, 0, 1) - grid_XY_vec.reshape(2, output_shape[0], output_shape[1])
