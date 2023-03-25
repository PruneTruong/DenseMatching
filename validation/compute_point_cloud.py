from PIL import Image
import numpy as np
import cv2
import os
from matplotlib import cm
cmap = cm.get_cmap("jet")
from matplotlib.colors import Normalize
import torch
try:
    import trimesh
except:
    print('didnt load trimesh')


def mesh_triangles(match, idx_mat):
    h, w = match.shape

    i, j = np.arange(h-1)[:, None], np.arange(1, w)[None, :]
    triangles_up = match[i, j-1] & match[i+1, j-1] & match[i, j]
    triangles_down = match[i+1, j] & match[i+1, j-1] & match[i, j]

    triangle_up_idx = np.stack([
        idx_mat[i, j - 1],
        idx_mat[i + 1, j - 1],
        idx_mat[i, j]
    ], axis=2)

    triangle_down_idx = np.stack([
        idx_mat[i + 1, j],
        idx_mat[i + 1, j - 1],
        idx_mat[i, j]
    ], axis=2)

    return np.concatenate(
       [triangle_up_idx[triangles_up], triangle_down_idx[triangles_down]],
        axis=0
    )


def compute_focal_lengths(fund, Issize, Itsize):
    F_tensor = torch.tensor(fund).float()
    
    f_t = torch.nn.Parameter(max(Issize) * torch.ones(1))
    f_s = torch.nn.Parameter(max(Itsize) * torch.ones(1))
    f_s.require_grad = True
    f_t.require_grad = True

    opt = torch.optim.SGD([f_s, f_t], lr=100, momentum=0.9)

    for lr in [1, .1]:
        opt.lr = lr
        for i in range(2000):
            K1 = torch.eye(3)
            K1[0, 0] = f_s
            K1[1, 1] = f_s

            K2 = torch.eye(3)
            K2[0, 0] = f_t
            K2[1, 1] = f_t

            E = K2.t() @ F_tensor @ K1
            u, s, v = torch.svd(E / torch.norm(E))
            l = s[0] - s[1]
            opt.zero_grad()
            l.backward()
            opt.step()

    return f_s.item(), f_t.item()


def get_rgba_im(depthmap):
    vmin, vmax = np.percentile(depthmap[depthmap > 0], [5, 95], )

    alphas = np.zeros(depthmap.shape)
    alphas[depthmap > 0] = 1.

    colors = Normalize(vmin, vmax, clip=True)(depthmap)
    colors = cmap(colors)

    colors[..., -1] = alphas
    return Image.fromarray((colors * 255).astype(np.uint8))


def get_point_cloud_color(source, target, pts1, pts2, by_path=True):

    imA = source
    imB = target

    npA = np.array(imA).astype(int)
    npB = np.array(imB).astype(int)

    # cA = npA[np.clip(pts1[:, 1].astype(int), 0, npA.shape[0] - 1),
    #          np.clip(pts1[:, 0].astype(int), 0, npA.shape[1] - 1)]
    cB = npB[np.clip(pts2[:, 1].astype(int), 0, npB.shape[0] - 1),
             np.clip(pts2[:, 0].astype(int), 0, npB.shape[1] - 1)]

    # return (cA + cB) / 2
    return cB


def compute_triangulation_angle(point_cloud, R, t):
    ray1 = point_cloud
    ray2 = point_cloud + (R.T @ t).T
        
    cos = np.sum(ray1 * ray2, axis=1) / np.linalg.norm(ray1, axis=1) / np.linalg.norm(ray2, axis=1)
    return np.arccos(cos) / np.pi * 180


def reproject_point_cloud(point_cloud, K, R=None, t=None):
    if R is None:
        R = np.eye(3)
        t = np.zeros((3, 1))
    # the transpose appear ebcause point_cloud (and image coordinates) are in the form Nx4/3 instead of 4xN.
    on_im_plane = (point_cloud @ R.T + t.T)    
    in_pixel = on_im_plane @ K.T
    return in_pixel[:, :2] / in_pixel[:, 2:], on_im_plane[:, 2]


def compute_Kmatrix(idx, K_list, org_imsizes, resized_shapes):
    ka = K_list[idx]

    ka[0, 2] = org_imsizes[idx][0] / 2
    ka[1, 2] = org_imsizes[idx][1] / 2

    rescale1 = np.diag([
        resized_shapes[idx][0] / org_imsizes[idx][0],
        resized_shapes[idx][1] / org_imsizes[idx][1],
        1.
    ])
    
    return rescale1 @ ka


def estimate_pose(normed_kpts0, normed_kpts1, K0, K1, ransac=True, thresh=1.0, conf=0.99999):
    if len(normed_kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    if ransac:
        E, mask = cv2.findEssentialMat(
            normed_kpts0, normed_kpts1, np.eye(3), threshold=norm_thresh,
            prob=conf,
            method=cv2.RANSAC)
    else:
        E, mask = cv2.findFundamentalMat(
            normed_kpts0, normed_kpts1,  method=cv2.FM_8POINT
        )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, normed_kpts0, normed_kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def compute_3D_pos(n_pts1, n_pts2, R, t):
    """
    Computes the 3D points from 2D matches using the Direct Linear Transform
    Args:
        n_pts1: points in the first image Nx2
        n_pts2: points in the second image Nx2
        R: relative rotation matrix of second image with respect to first (3x3)
        t: relative translation vector of second image with respect to first (3x1)

    Returns:
        coord: 3D coordinates corresponding to the points, Nx3
    """
    P1 = np.concatenate((
        np.eye(3),
        np.zeros((3, 1))
    ), axis=1)  # 3 * 4

    P2 = np.concatenate((R, t), axis=1)  # 3 * 4

    # build A matrix of shape N * 4 * 4
    A = np.stack((
        n_pts1[:, 0:1] * P1[2, :] - P1[0, :],
        n_pts1[:, 1:2] * P1[2, :] - P1[1, :],
        n_pts2[:, 0:1] * P2[2, :] - P2[0, :],
        n_pts2[:, 1:2] * P2[2, :] - P2[1, :]
    ), axis=1)

    _, _, V = np.linalg.svd(A)

    X = V[:, -1, :]  # N * 4

    coord = X[:, :3] / X[:, 3:4]

    '''
    # triangulate points

    pts4D = cv2.triangulatePoints(P1, P2, n_pts1, n_pts2).T

    # convert from homogeneous coordinates to 3D
    pts3D = pts4D[:, :3] / pts4D[:, 3, None]

    # plot with matplotlib
    Xs, Zs, Ys = [pts3D[:, i] for i in range(3)]
    '''
    return coord


def image_to_numpy_array(img):
    if not isinstance(img, np.ndarray):
        img = img.cpu().numpy()
    if len(img.shape) == 4:
        img = img[0]
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    return img


def compute_and_save(pts1, pts2, Is, It, savepath, thresholds=None, K1=None, K2=None, R=None, t=None, suffix="",
                     mask=None, use_inliers_of_essentiel_ransac=False):
    """

    Args:
        pts1: points in the first image Nx2
        pts2: points in the second image Nx2
        Is: first image
        It: second image
        savepath: path to directory where to save the reconstruction .ply files
        thresholds: dict with fields 'fundamental_ransac' and 'essential_ransac' as thresholds
        K1: intrinsic matrix for first image
        K2: intrinsic matrix for second image
        R: relative rotation matrixof second image with respect to first image
        t: relative translation vector of second image with respect to first image
        suffix: name of particular reconstruction file
        mask: inlier mask outputted by RANSAC when computing the essential matrix

    Returns:

    """
    default_thresholds = {"fundamental_ransac": 1.0, "essential_ransac": 1.0}
    if thresholds is not None:
        default_thresholds.update(thresholds)
    # put images to correct size
    Is = image_to_numpy_array(Is)
    It = image_to_numpy_array(It)
    # final shape should be H, W, 3

    fund_ransac_thresh = default_thresholds["fundamental_ransac"]
    ess_ransac_thresh = default_thresholds["essential_ransac"]
    Isw, Ish = Is.shape[:2]
    Itw, Ith = It.shape[:2]

    if K1 is None or K2 is None:
        resF = cv2.findFundamentalMat(pts1, pts2, ransacReprojThreshold=fund_ransac_thresh, method=cv2.FM_RANSAC)

        # f_s, f_t = compute_focal_lengths(resF[0], (Isw, Ish), (Itw, Ith))
        f_t = max(Isw, Ish)
        f_s = max(Itw, Ith)

    norm_pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    norm_pts2 = (pts2 - K2[[0, 1], [2, 2]][None]) / K2[[0, 1], [0, 1]][None]

    if R is None or t is None:
        # find essential matrix and decompose into rotation and translation
        (R, t, mask) = estimate_pose(norm_pts1, norm_pts2, K1, K2, ransac=True)

    if use_inliers_of_essentiel_ransac:
        if mask is not None:
            norm_pts1 = norm_pts1[mask]
            norm_pts2 = norm_pts2[mask]
            pts1 = pts1[mask]
            pts2 = pts2[mask]

    # do triangulation to obtain the 3D points corresponding to the matches between the images
    coords = compute_3D_pos(norm_pts1, norm_pts2, R, t.reshape(-1, 1))

    est_pts1, depth1 = reproject_point_cloud(coords, K1)
    est_pts2, depth2 = reproject_point_cloud(coords, K2, R, t)

    int_pts1 = np.rint(pts1).astype(int)
    int_pts2 = np.rint(pts2).astype(int)

    depthmap1 = -1 * np.ones((Ish, Isw))
    depthmap2 = -1 * np.ones((Ith, Itw))

    depthmap1[np.clip(int_pts1[:, 1], 0, Ish - 1), np.clip(int_pts1[:, 0], 0, Isw - 1)] = depth1
    depthmap2[np.clip(int_pts2[:, 1], 0, Ith - 1), np.clip(int_pts2[:, 0], 0, Itw - 1)] = depth2

    # plot the depth
    # imA = get_rgba_im(depthmap1)
    # imB = get_rgba_im(depthmap2)

    # imA.save(os.path.join(savepath, "depthmapSource_{}.png".format(suffix)))
    # imB.save(os.path.join(savepath, "depthmapTarget_{}.png".format(suffix)))

    # get the RGB colors of the points
    colors = get_point_cloud_color(Is, It, pts1, pts2, by_path=False)

    # create the mesh with the 3D coordinates and the corresponding RGB
    mesh = trimesh.Trimesh(
        vertices=coords,
        vertex_colors=colors
    )

    # some thresholding, I don't think this is really needed
    max_depth = np.percentile(coords[:, 2], 99)
    min_depth = np.percentile(coords[:, 2], 1)
    mesh.update_vertices(mesh.vertices[:, 2] > min_depth)
    mesh.update_vertices(mesh.vertices[:, 2] < max_depth)
    # mesh.update_faces(mesh.area_faces < thresholds["max_triangle_size"])
    mesh.export(os.path.join(savepath,   "mesh_{}.ply".format(suffix)))


