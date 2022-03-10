# extracted from NC-Net
import torch
import numpy as np
from skimage import draw


def poly_to_mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def poly_str_to_mask(polygon_x, polygon_y, out_h, out_w):
    mask_np = poly_to_mask(vertex_col_coords=polygon_x,
                           vertex_row_coords=polygon_y, shape=[out_h, out_w])
    mask = torch.FloatTensor(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    mask = mask.cuda()
    return mask_np, mask


def intersection_over_union(warped_mask, target_mask):
    relative_part_weight = torch.sum(torch.sum(target_mask.data.gt(0.5).float(), 2, True), 3, True) / \
                           torch.sum(target_mask.data.gt(0.5).float())
    part_iou = torch.sum(torch.sum((warped_mask.data.gt(0.5) & target_mask.data.gt(0.5)).float(), 2, True), 3, True) \
               / torch.sum(torch.sum((warped_mask.data.gt(0.5) | target_mask.data.gt(0.5)).float(), 2, True), 3, True)
    weighted_iou = torch.sum(torch.mul(relative_part_weight, part_iou))
    return weighted_iou


def label_transfer_accuracy(warped_mask, target_mask):
    return torch.mean((warped_mask.data.gt(0.5) == target_mask.data.gt(0.5)).double())


def localization_error(source_mask_np, target_mask_np, flow_np):
    """
    Args:
        source_mask_np:
        target_mask_np:
        flow_np: shape H x W x 2
    """
    h_tgt, w_tgt = target_mask_np.shape[:2]
    h_src, w_src = source_mask_np.shape[:2]

    # initial pixel positions x1,y1 in target image
    x1, y1 = np.meshgrid(range(0, w_tgt), range(0, h_tgt))
    # sampling pixel positions x2,y2
    x2 = x1 + flow_np[:, :, 0]
    y2 = y1 + flow_np[:, :, 1]

    # compute in-bound coords for each image
    in_bound = (x2 >= 0) & (x2 < w_src) & (y2 >= 0) & (y2 < h_src)
    row, col = np.where(in_bound)
    row_1 = y1[row, col].flatten().astype(np.int)
    col_1 = x1[row, col].flatten().astype(np.int)
    row_2 = y2[row, col].flatten().astype(np.int)
    col_2 = x2[row, col].flatten().astype(np.int)

    # compute relative positions
    target_loc_x, target_loc_y = obj_ptr(target_mask_np)
    source_loc_x, source_loc_y = obj_ptr(source_mask_np)
    x1_rel = target_loc_x[row_1, col_1]
    y1_rel = target_loc_y[row_1, col_1]
    x2_rel = source_loc_x[row_2, col_2]
    y2_rel = source_loc_y[row_2, col_2]

    # compute localization error
    loc_err = np.mean(np.abs(x1_rel - x2_rel) + np.abs(y1_rel - y2_rel))

    return loc_err


def obj_ptr(mask):
    # computes images of normalized coordinates around bounding box
    # kept function name from DSP code
    h, w = mask.shape[0], mask.shape[1]
    y, x = np.where(mask > 0.5)
    left = np.min(x)
    right = np.max(x)
    top = np.min(y)
    bottom = np.max(y)
    fg_width = right - left + 1
    fg_height = bottom - top + 1
    x_image, y_image = np.meshgrid(range(1, w + 1), range(1, h + 1))
    x_image = (x_image - left) / fg_width
    y_image = (y_image - top) / fg_height
    return x_image, y_image


