import numpy as np
import cv2
import torch
import torch.nn as nn
from packaging import version


def remap_using_flow_fields(image, disp_x, disp_y, interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT):
    """
    Opencv remap
    map_x contains the index of the matching horizontal position of each pixel [i,j] while map_y contains the
    index of the matching vertical position of each pixel [i,j]

    All arrays are numpy
    args:
        image: image to remap, HxWxC
        disp_x: displacement in the horizontal direction to apply to each pixel. must be float32. HxW
        disp_y: displacement in the vertical direction to apply to each pixel. must be float32. HxW
        interpolation
        border_mode
    output:
        remapped image. HxWxC
    """
    h_scale, w_scale=disp_x.shape[:2]

    # estimate the grid
    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    map_x = (X+disp_x).astype(np.float32)
    map_y = (Y+disp_y).astype(np.float32)
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)

    return remapped_image


def remap_using_correspondence_map(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                   border_mode=cv2.BORDER_CONSTANT):
    """
    Opencv remap
    map_x contains the index of the matching horizontal position of each pixel [i,j] while map_y contains the
    index of the matching vertical position of each pixel [i,j]

    All arrays are numpy
    args:
        image: image to remap, HxWxC
        map_x: mapping in the horizontal direction to apply to each pixel. must be float32. HxW
        map_y: mapping in the vertical direction to apply to each pixel. must be float32. HxW
        interpolation
        border_mode
    output:
        remapped image. HxWxC
    """
    remapped_image = cv2.remap(image, map_x, map_y, interpolation=interpolation, borderMode=border_mode)
    return remapped_image


def warp(x, flo, padding_mode='zeros', return_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    args:
        x: [B, C, H, W]
        flo: [B, 2, H, W] flow
    outputs:
        output: warped x [B, C, H, W]
    """
    B, C, H, W = flo.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo
    # makes a mapping out of the flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)

    if version.parse(torch.__version__) >= version.parse("1.3"):
        output = nn.functional.grid_sample(x, vgrid, align_corners=True, padding_mode=padding_mode)
    else:
        output = nn.functional.grid_sample(x, vgrid, padding_mode=padding_mode)

    if return_mask:
        vgrid = vgrid.permute(0, 3, 1, 2)
        mask = (vgrid[:, 0] > -1) & (vgrid[:, 1] > -1) & (vgrid[:, 0] < 1) & (vgrid[:, 1] < 1)
        return output, mask
    return output


def warp_with_mapping(x, vgrid, padding_mode='zeros', return_mask=False):
    """
    warp an image/tensor (im2) back to im1, according to the mapping (in pixel coordinates)

    args:
        x: [B, C, H, W] (im2)
        vgrid: [B, 2, H, W] mapping instead of flow
    outputs:
        output: warped x [B, C, H, W]
    """
    B, C, H, W = x.size()
    # mesh grid
    vgrid = vgrid.clone()
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    if version.parse(torch.__version__) >= version.parse("1.3"):
        output = nn.functional.grid_sample(x, vgrid, align_corners=True, padding_mode=padding_mode)
    else:
        output = nn.functional.grid_sample(x, vgrid, padding_mode=padding_mode)

    if return_mask:
        vgrid = vgrid.permute(0, 3, 1, 2)
        mask = (vgrid[:, 0] > -1) & (vgrid[:, 1] > -1) & (vgrid[:, 0] < 1) & (vgrid[:, 1] < 1)
        return output, mask
    return output
