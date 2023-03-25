# Video's features
import numpy as np
from sklearn.decomposition import PCA
import cv2
import imageio as io
import visdom
import torchvision
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn.functional as F
from numpy import unravel_index


def plot_correlation_for_probabilistic_warpc(target_image, source_image, target_image_prime, flow_gt, correlation_volume_tp_to_s,
                                             correlation_volume_s_to_t, correlation_volume_tp_to_t, correlation_volume_t_to_s, save_path, name,
                                             normalization='relu_l2norm', mask=None, plot_individual=False, mask_top=None):
    """
    Args:
        target_image: 3xHxW
        source_image: 3xHxW
        correlation: (HxW)xH_c x W_c
    """
    # choose 10 random points
    _, H_ori, W_ori = flow_gt.shape
    _, H, W = correlation_volume_s_to_t.shape

    nbr_pts_per_row = 2

    plot_occ = False
    if correlation_volume_tp_to_s.shape[0] == H*W + 1:
        plot_occ = True
        occ_mask_tp_to_t = correlation_volume_tp_to_t.permute(1, 2, 0)[:, :, -1].cpu().numpy()
    correlation_volume_tp_to_s = correlation_volume_tp_to_s[:H*W]
    correlation_volume_s_to_t = correlation_volume_s_to_t[:H*W]
    correlation_volume_tp_to_t = correlation_volume_tp_to_t[:H*W]
    correlation_volume_t_to_s = correlation_volume_t_to_s[:H*W]

    if mask is not None:
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), (H, W), mode='bilinear', align_corners=False) \
            .squeeze().cpu().numpy()
        kp_in_mask = np.where(mask)
        n = kp_in_mask[0].shape[0]
        kp_in_mask = np.concatenate([kp_in_mask[1].reshape(n, 1), kp_in_mask[0].reshape(n, 1)], axis=1)
        selection = np.int32(np.linspace(0, n-1, nbr_pts_per_row*2))
        X = kp_in_mask[selection, 0]
        Y = kp_in_mask[selection, 1]
    else:
        X, Y = np.meshgrid(np.arange(W // 5, W - 1, W // nbr_pts_per_row),
                           np.arange(H // 5, H - 1, H // nbr_pts_per_row))
        X = np.int32(X.flatten())
        Y = np.int32(Y.flatten())

    mean_values = torch.tensor([0.485, 0.456, 0.406],
                               dtype=source_image.dtype).view(3, 1, 1)
    std_values = torch.tensor([0.229, 0.224, 0.225],
                              dtype=source_image.dtype).view(3, 1, 1)

    if plot_individual:
        cv2.imwrite('{}/{}_source.jpg'.format(save_path, name),
                    ((source_image.squeeze(0).cpu() * std_values + mean_values).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        cv2.imwrite('{}/{}_target.jpg'.format(save_path, name),
                    ((target_image.squeeze(0).cpu() * std_values + mean_values).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        cv2.imwrite('{}/{}_target_prime.jpg'.format(save_path, name),
                    ((target_image_prime.squeeze(0).cpu() * std_values + mean_values).permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    # resizing of source and target image to correlation size
    image_source = F.interpolate(source_image.unsqueeze(0).cpu() * std_values +
                                 mean_values, (H, W), mode='area').squeeze().permute(1, 2, 0).numpy()
    image_target = F.interpolate(target_image.unsqueeze(0).cpu() * std_values +
                                 mean_values, (H, W), mode='area').squeeze().permute(1, 2, 0).numpy()
    image_target_prime = F.interpolate(target_image_prime.unsqueeze(0).cpu() * std_values +
                                       mean_values, (H, W), mode='area').squeeze().permute(1, 2, 0).numpy()
    flow_gt_resized = F.interpolate(flow_gt.unsqueeze(0), (H, W), mode='bilinear', align_corners=False) \
        .squeeze(0).permute(1, 2, 0).cpu().numpy()
    flow_gt_resized[:, :, 0] *= float(W) / float(W_ori)
    flow_gt_resized[:, :, 1] *= float(H) / float(H_ori)

    if mask is not None:
        fig, axis = plt.subplots(len(X) + 1, 7, figsize=(20, 20))
        axis[len(X)][0].imshow(mask, vmin=0, vmax=1.0)
        axis[len(X)][1].axis('off')
        if plot_occ:
            axis[len(X)][1].imshow(occ_mask_tp_to_t, vmin=0, vmax=1.0)
            axis[len(X)][1].set_title('Occlusion mask, \n min={:.5f}, max={:.5f}'.format(occ_mask_tp_to_t.min(),
                                                                                         occ_mask_tp_to_t.max()))
            axis[len(X)][1].axis('off')
        if mask_top is not None:
            axis[len(X)][2].imshow(mask_top.cpu().numpy(), vmin=0, vmax=1.0)
            axis[len(X)][2].set_title('mask top percent')
            axis[len(X)][2].axis('off')

    else:
        fig, axis = plt.subplots(len(X), 7, figsize=(20, 20))
    for i in range(len(X)):
        pt = [X[i], Y[i]]
        # first coordinate is horizontal, second is vertical

        # point in the target prime image
        gt_point_in_target = [int(round(pt[0] + flow_gt_resized[int(pt[1]), int(pt[0]), 0])),
                        int(round(pt[1] + flow_gt_resized[int(pt[1]), int(pt[0]), 1]))]


        correlation_at_point_tp_to_s = correlation_volume_tp_to_s.permute(1, 2, 0).view(H, W, H, W)[pt[1], pt[0]].cpu().numpy()
        max_pt_in_source = unravel_index(correlation_at_point_tp_to_s.argmax(), correlation_at_point_tp_to_s.shape)
        max_pt_in_source = [max_pt_in_source[1], max_pt_in_source[0]]


        correlation_at_point = correlation_volume_tp_to_t.permute(1, 2, 0).view(H, W, H, W)[pt[1], pt[0]].cpu().numpy()
        max_pt_in_target_direct = unravel_index(correlation_at_point.argmax(), correlation_at_point.shape)
        max_pt_in_target_direct = [max_pt_in_target_direct[1], max_pt_in_target_direct[0]]

        # target image prime
        axis[i][0].imshow(np.clip(image_target_prime, 0, 1), vmin=0, vmax=1.0)
        axis[i][0].scatter(pt[0], pt[1], s=8, color='red')
        axis[i][0].set_title('target image prime, start pt in red')

        axis[i][3].imshow(correlation_at_point)

        if not (max_pt_in_target_direct[0] > W or max_pt_in_target_direct[0] < 0 or max_pt_in_target_direct[1] > H or max_pt_in_target_direct[1] < 0):
            axis[i][3].set_title('P tp to t, \nmin={:.5f}, max={:.5f}\n sum={:.4f}'
                                 .format(correlation_at_point.min(), correlation_at_point.max(),
                                         correlation_at_point.sum()))
            axis[i][2].scatter(max_pt_in_target_direct[0], max_pt_in_target_direct[1], s=8, color='red')
            axis[i][3].scatter(max_pt_in_target_direct[0], max_pt_in_target_direct[1], s=8, color='red')
        else:
            axis[i][3].set_title('P tp to t, est pt is outside')

        # source
        axis[i][1].imshow(np.clip(image_source, 0, 1), vmin=0, vmax=1.0)

        axis[i][4].imshow(correlation_at_point_tp_to_s)
        axis[i][4].set_title('P tp to s,\n min={:.5f}, \nmax={:.5f}\n sum={:.4f}'
                             .format(correlation_at_point_tp_to_s.min(), correlation_at_point_tp_to_s.max(),
                                     correlation_at_point_tp_to_s.sum()))

        if not (max_pt_in_source[0] > W or max_pt_in_source[0] < 0 or max_pt_in_source[1] > H or max_pt_in_source[1] < 0):
            axis[i][1].scatter(max_pt_in_source[0], max_pt_in_source[1], s=8, color='blue')
            axis[i][4].scatter(max_pt_in_source[0], max_pt_in_source[1], s=8, color='blue')
            axis[i][1].set_title('source_image')

            correlation_at_point_s_to_t = correlation_volume_s_to_t.permute(1, 2, 0).view(H, W, H, W)[
                max_pt_in_source[1], max_pt_in_source[0]].cpu().numpy()

            axis[i][5].imshow(correlation_at_point_s_to_t)
            axis[i][5].set_title('P s to t for point in source\n min={:.5f}, \nmax={:.5f}\n sum={:.4f}'.
                                 format(correlation_at_point_s_to_t.min(),
                                        correlation_at_point_s_to_t.max(),
                                 correlation_at_point_s_to_t.sum()))

            max_pt_in_target = unravel_index(correlation_at_point_s_to_t.argmax(), correlation_at_point_s_to_t.shape)
            max_pt_in_target = [max_pt_in_target[1], max_pt_in_target[0]]

            if not (max_pt_in_target[0] > W or max_pt_in_target[0] < 0 or max_pt_in_target[1] > H or max_pt_in_target[1] < 0):
                axis[i][2].scatter(max_pt_in_target[0], max_pt_in_target[1], s=8, color='blue')
                axis[i][5].scatter(max_pt_in_target[0], max_pt_in_target[1], s=8, color='blue')

        else:
            axis[i][1].set_title('source_image, point is outside')
        # target
        axis[i][2].imshow(image_target, vmin=0, vmax=1.0)

        if gt_point_in_target[0] > W or gt_point_in_target[0] < 0 or gt_point_in_target[1] > H or gt_point_in_target[1] < 0:
            axis[i][2].set_title('Target image, gt pt outside (green)')
        else:
            axis[i][2].scatter(gt_point_in_target[0], gt_point_in_target[1], s=8, color='green')
            axis[i][3].scatter(gt_point_in_target[0], gt_point_in_target[1], s=8, color='green')
            axis[i][5].scatter(gt_point_in_target[0], gt_point_in_target[1], s=8, color='green')
            axis[i][2].set_title('target image, gt (green), \npt from tp to t (red), \nfrom compo (blue)')

        correlation_at_point_t_to_s = correlation_volume_t_to_s.permute(1, 2, 0).view(H, W, H, W)[pt[1], pt[0]].cpu().numpy()
        axis[i][6].imshow(correlation_at_point_t_to_s)
        axis[i][6].set_title('A t to s, \nmin={:.4f}, \nmax={:.4f}\n sum={:.4f}'
                             .format(correlation_at_point_t_to_s.min(),
                                     correlation_at_point_t_to_s.max(), correlation_at_point_t_to_s.sum()))

    fig.tight_layout()
    fig.savefig('{}/{}.png'.format(save_path, name),
                bbox_inches='tight')
    plt.close(fig)


def pca_feats(ff, K=1, solver='auto', whiten=True, img_normalize=True):
    ## expect ff to be   N x C x H x W

    N, C, H, W = ff.shape
    pca = PCA(
        n_components=3 * K,
        svd_solver=solver,
        whiten=whiten
    )

    ff = ff.transpose(1, 2).transpose(2, 3)
    ff = ff.reshape(N * H * W, C).numpy()

    pca_ff = torch.Tensor(pca.fit_transform(ff))
    pca_ff = pca_ff.view(N, H, W, 3 * K)
    pca_ff = pca_ff.transpose(3, 2).transpose(2, 1)

    pca_ff = [pca_ff[:, kk:kk + 3] for kk in range(0, pca_ff.shape[1], 3)]

    if img_normalize:
        pca_ff = [(x - x.min()) / (x.max() - x.min()) for x in pca_ff]

    return pca_ff[0] if K == 1 else pca_ff


def make_gif(video, outname='/tmp/test.gif', sz=256):
    if hasattr(video, 'shape'):
        video = video.cpu()
        if video.shape[0] == 3:
            video = video.transpose(0, 1)

        video = video.numpy().transpose(0, 2, 3, 1)
        video = (video * 255).astype(np.uint8)

    video = [cv2.resize(vv, (sz, sz)) for vv in video]

    if outname is None:
        return np.stack(video)

    io.mimsave(outname, video, duration=0.2)


def draw_matches(x1, x2, i1, i2):
    # x1, x2 = f1, f2/
    detach = lambda x: x.detach().cpu().numpy().transpose(1, 2, 0) * 255
    i1, i2 = detach(i1), detach(i2)
    i1, i2 = cv2.resize(i1, (400, 400)), cv2.resize(i2, (400, 400))

    for check in [True]:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=check)
        # matches = bf.match(x1.permute(0,2,1).view(-1, 128).cpu().detach().numpy(), x2.permute(0,2,1).view(-1, 128).cpu().detach().numpy())

        h = int(x1.shape[-1] ** 0.5)
        matches = bf.match(x1.t().cpu().detach().numpy(), x2.t().cpu().detach().numpy())

        scale = i1.shape[-2] / h
        grid = torch.stack([torch.arange(0, h)[None].repeat(h, 1), torch.arange(0, h)[:, None].repeat(1, h)])

        grid = grid.view(2, -1)
        grid = grid * scale + scale // 2

        kps = [cv2.KeyPoint(grid[0][i], grid[1][i], 1) for i in range(grid.shape[-1])]

        matches = sorted(matches, key=lambda x: x.distance)

        # img1 = img2 = np.zeros((40, 40, 3))
        out = cv2.drawMatches(i1.astype(np.uint8), kps, i2.astype(np.uint8), kps, matches[:], None, flags=2).transpose(
            2, 0, 1)

    return out


import wandb


class Visualize(object):
    def __init__(self, args):
        self._env_name = args.name
        self.vis = visdom.Visdom(
            port=args.port,
            server='http://%s' % args.server,
            env=self._env_name,
        )
        self.args = args

        self._init = False

    def wandb_init(self, model):
        if not self._init:
            self._init = True
            wandb.init(project="videowalk", group="release", config=self.args)
            wandb.watch(model)

    def log(self, key_vals):
        return wandb.log(key_vals)

    def nn_patches(self, P, A_k, prefix='', N=10, K=20):
        nn_patches(self.vis, P, A_k, prefix, N, K)

    def save(self):
        self.vis.save([self._env_name])


def get_stride(im_sz, p_sz, res):
    stride = (im_sz - p_sz) // (res - 1)
    return stride


def nn_patches(vis, P, A_k, prefix='', N=10, K=20):
    # produces nearest neighbor visualization of N patches given an affinity matrix with K channels

    P = P.cpu().detach().numpy()
    P -= P.min()
    P /= P.max()

    A_k = A_k.cpu().detach().numpy()  # .transpose(-1,-2).numpy()
    # assert np.allclose(A_k.sum(-1), 1)

    A = np.sort(A_k, axis=-1)
    I = np.argsort(-A_k, axis=-1)

    vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header' % (prefix))

    for n, i in enumerate(np.random.permutation(P.shape[0])[:N]):
        p = P[i]
        vis.text('', opts=dict(width=10000, height=1), win='%s_patch_header_%s' % (prefix, n))
        # vis.image(p,  win='%s_patch_query_%s' % (prefix, n))

        for k in range(I.shape[0]):
            vis.images(P[I[k, i, :K]], nrow=min(I.shape[-1], 20), win='%s_patch_values_%s_%s' % (prefix, n, k))
            vis.bar(A[k, i][::-1][:K], opts=dict(height=150, width=500), win='%s_patch_affinity_%s_%s' % (prefix, n, k))


def compute_flow(corr):
    # that is actually mapping
    # assume batched affinity, shape N x H * W x H x W
    h = w = int(corr.shape[-1] ** 0.5)

    # x1 -> x2
    corr = corr.view(*corr.shape[:-1], h, w)
    nnf = corr.argmax(dim=1)

    u = nnf % w  # nnf.shape[-1]
    v = nnf / h  # nnf.shape[-2] # nnf is an IntTensor so rounds automatically

    rr = torch.arange(u.shape[-1])[None].long().cuda()

    for i in range(u.shape[-1]):
        u[:, i] -= rr

    for i in range(v.shape[-1]):
        v[:, :, i] -= rr

    return u, v


def vis_flow_plt(u, v, x1, x2, A):
    flows = torch.stack([u, v], dim=-1).cpu().numpy()
    I, flows = x1.cpu().numpy(), flows[0]

    H, W = flows.shape[:2]
    Ih, Iw, = I.shape[-2:]
    mx, my = np.mgrid[0:Ih:Ih / (H + 1), 0:Iw:Iw / (W + 1)][:, 1:, 1:]
    skip = (slice(None, None, 1), slice(None, None, 1))

    ii = 0
    fig, ax = plt.subplots()
    im = ax.imshow((I.transpose(1, 2, 0)), )

    C = cm.jet(torch.nn.functional.softmax((A * A.log()).sum(-1).cpu(), dim=-1))
    ax.quiver(my[skip], mx[skip], flows[..., 0][skip], flows[..., 1][skip] * -1, C)  # , scale=1, scale_units='dots')
    # ax.quiver(mx[skip], my[skip], flows[...,0][skip], flows[...,1][skip])

    return plt


def frame_pair(x1, x2, f1, f2, A, xent_loss, viz):
    normalize = lambda xx: (xx - xx.min()) / (xx - xx.min()).max()
    spatialize = lambda xx: xx.view(*xx.shape[:-1], int(xx.shape[-1] ** 0.5), int(xx.shape[-1] ** 0.5))

    N = A.shape[-1]
    H = W = int(N ** 0.5)
    AA = A.view(-1, H * W, H, W)

    ##############################################
    ## Visualize PCA of Embeddings, Correspondences
    ##############################################

    # Single image input, no patches
    # X here is B x C x H x W
    x1, x2 = normalize(x1[0]), normalize(x2[0])
    f1, f2 = f1[0], f2[0]
    ff1, ff2 = spatialize(f1), spatialize(f2)

    xx = torch.stack([x1, x2]).detach().cpu()
    viz.images(xx, win='imgs')

    # Flow
    u, v = compute_flow(A)
    flow_plt = vis_flow_plt(u, v, x1, x2, A)
    viz.matplot(flow_plt, win='flow_quiver')

    # Keypoint Correspondences
    kp_corr = draw_matches(f1, f2, x1, x2)
    viz.image(kp_corr, win='kpcorr')

    '''
    # # PCA VIZ
    pca_ff = pca_feats(torch.stack([ff1, ff2]).detach().cpu())
    pca_ff = make_gif(pca_ff, outname=None)
    viz.images(pca_ff.transpose(0, -1, 1, 2), win='pcafeats', opts=dict(title=f"{t1} {t2}"))
    '''

    ##############################################
    # LOSS VIS
    ##############################################
    color = cm.get_cmap('winter')

    xx = normalize(xent_loss[:H * W])
    img_grid = [cv2.resize(aa, (50, 50), interpolation=cv2.INTER_NEAREST)[None]
                for aa in AA[0, :, :, :, None].cpu().detach().numpy()]
    img_grid = [img_grid[_].repeat(3, 0) * np.array(color(xx[_].item()))[:3, None, None] for _ in range(H * W)]
    img_grid = [img_grid[_] / img_grid[_].max() for _ in range(H * W)]
    img_grid = torch.from_numpy(np.array(img_grid))
    img_grid = torchvision.utils.make_grid(img_grid, nrow=H, padding=1, pad_value=1)

    # img_grid = cv2.resize(img_grid.permute(1, 2, 0).cpu().detach().numpy(), (1000, 1000), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    viz.images(img_grid, win='lossvis')