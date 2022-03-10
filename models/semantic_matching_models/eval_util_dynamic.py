import torch
import torch.nn
from torch.autograd import Variable
import numpy as np


def normalize_axis(x, L):
    return (x - 1 - (L - 1) / 2) * 2 / (L - 1)


def unnormalize_axis(x, L):
    return x * (L - 1) / 2 + 1 + (L - 1) / 2


def mergeA_mergeB_from_out(out):
    # corr_out_set = out['corr'] #(B,S,1,Ha,Wa,Hb,Wb)
    # A_scaleatts_set = out['scaleatt']['A'] #(B,S,Ha,Wa)
    # B_scaleatts_set = out['scaleatt']['B'] #(B,S,Hb,Wb)

    corr_out_set, A_scaleatts_set, B_scaleatts_set = out
    B, S, _, Ha, Wa, Hb, Wb = corr_out_set.shape

    score_pos_overscales = []
    M_A_norm_overscales = []
    M_B_norm_overscales = []
    for si in range(S):  # iterate over scales
        corr_out_si = corr_out_set[:, si, :, :, :, :, :]
        score_pos_si, M_A_norm, M_B_norm = score_for_single_corr4d(corr4d=corr_out_si)  # MA_norm:(B,LB,Ha,Wa)

        # add
        score_pos_overscales.append(score_pos_si)
        M_A_norm_overscales.append(M_A_norm)
        M_B_norm_overscales.append(M_B_norm)

    M_A_norm_overscales = torch.stack(M_A_norm_overscales, dim=1)  # (B,LB,Ha,Wa)->(B,S,LB,Ha,Wa)
    M_B_norm_overscales = torch.stack(M_B_norm_overscales, dim=1)  # (B,LA,Hb,Wb)->(B,S,LA,Hb,Wb)

    # merge scoremap using atts
    MergedA = torch.sum(M_A_norm_overscales * A_scaleatts_set.view(B, S, 1, Ha, Wa), dim=1)  # (B,LB,Ha,Wa)
    MergedB = torch.sum(M_B_norm_overscales * B_scaleatts_set.view(B, S, 1, Hb, Wb), dim=1)  # (B,LA,Hb,Wb)

    return MergedA, MergedB


def corr_to_matches(out, delta4d=None, k_size=1, do_softmax=False, scale='centered', return_indices=False,
                    invert_matching_direction=False):
    MergeA, MergeB = mergeA_mergeB_from_out(out)  # MergeA (B,LB,Ha,Wa), MergeB (B,LA,Hb,Wb)

    # to_cuda = lambda x: x.cuda() if corr4d.is_cuda else x
    # batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()

    to_cuda = lambda x: x.cuda() if out[0].is_cuda else x
    batch_size, ch, _, fs1, fs2, fs3, fs4 = out[0].size()  # todo merge ch with S dimension

    if scale == 'centered':
        XA, YA = np.meshgrid(np.linspace(-1, 1, fs2 * k_size), np.linspace(-1, 1, fs1 * k_size))
        XB, YB = np.meshgrid(np.linspace(-1, 1, fs4 * k_size), np.linspace(-1, 1, fs3 * k_size))
    elif scale == 'positive':
        XA, YA = np.meshgrid(np.linspace(0, 1, fs2 * k_size), np.linspace(0, 1, fs1 * k_size))
        XB, YB = np.meshgrid(np.linspace(0, 1, fs4 * k_size), np.linspace(0, 1, fs3 * k_size))

    JA, IA = np.meshgrid(range(fs2), range(fs1))
    JB, IB = np.meshgrid(range(fs4), range(fs3))

    XA, YA = Variable(to_cuda(torch.FloatTensor(XA))), Variable(to_cuda(torch.FloatTensor(YA)))
    XB, YB = Variable(to_cuda(torch.FloatTensor(XB))), Variable(to_cuda(torch.FloatTensor(YB)))

    JA, IA = Variable(to_cuda(torch.LongTensor(JA).view(1, -1))), Variable(to_cuda(torch.LongTensor(IA).view(1, -1)))
    JB, IB = Variable(to_cuda(torch.LongTensor(JB).view(1, -1))), Variable(to_cuda(torch.LongTensor(IB).view(1, -1)))

    if invert_matching_direction:
        # nc_A_Bvec=corr4d.view(batch_size,fs1,fs2,fs3*fs4) #(B,Ha,Wa,LB)
        nc_A_Bvec = MergeA
        if do_softmax:
            nc_A_Bvec = torch.nn.functional.softmax(nc_A_Bvec, dim=1)

        match_A_vals, idx_A_Bvec = torch.max(nc_A_Bvec, dim=1)
        score = match_A_vals.view(batch_size, -1)

        iB = IB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        jB = JB.view(-1)[idx_A_Bvec.view(-1)].view(batch_size, -1)
        iA = IA.expand_as(iB)
        jA = JA.expand_as(jB)

    else:
        # nc_B_Avec=corr4d.view(batch_size,fs1*fs2,fs3,fs4) # [batch_idx,k_A,i_B,j_B] #(B,LA,Hb,Wb)
        nc_B_Avec = MergeB
        if do_softmax:
            nc_B_Avec = torch.nn.functional.softmax(nc_B_Avec, dim=1)

        match_B_vals, idx_B_Avec = torch.max(nc_B_Avec, dim=1)
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
    xB = XB[iB.view(-1), jB.view(-1)].view(batch_size, -1)
    yB = YB[iB.view(-1), jB.view(-1)].view(batch_size, -1)

    if return_indices:
        return (xA, yA, xB, yB, score, iA, jA, iB, jB)
    else:
        return (xA, yA, xB, yB, score)


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


def pck(source_points, warped_points, L_pck, alpha=0.1):
    # compute precentage of correct keypoints
    batch_size = source_points.size(0)
    pck = torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i, :]
        p_wrp = warped_points[i, :]
        N_pts = torch.sum(torch.ne(p_src[0, :], -1) * torch.ne(p_src[1, :], -1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:, :N_pts] - p_wrp[:, :N_pts], 2), 0), 0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance, L_pck_mat * alpha)
        pck[i] = torch.mean(correct_points.float())
    return pck


def pck_metric(batch, batch_start_idx, matches, stats, args, use_cuda=True, alpha=0.1):
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']

    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points, target_im_size)

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches, target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm, source_im_size)

    L_pck = batch['L_pck'].data

    current_batch_size = batch['source_im_size'].size(0)
    indices = range(batch_start_idx, batch_start_idx + current_batch_size)

    # compute PCK
    pck_batch = pck(source_points.data, warped_points.data, L_pck, alpha=alpha)
    stats['point_tnf']['pck'][indices] = pck_batch.unsqueeze(1).cpu().numpy()

    return stats
