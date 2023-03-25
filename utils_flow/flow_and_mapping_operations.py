import numpy as np
from packaging import version
import torch


from utils_flow.pixel_wise_mapping import remap_using_correspondence_map


def create_border_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image), and convert to float. """
    return get_gt_correspondence_mask(flow).float()


def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if isinstance(mapping, np.ndarray):
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[:, 0] >= 0, mapping[:, 0] <= w-1)
            mask_y = np.logical_and(mapping[:, 1] >= 0, mapping[:, 1] <= h-1)
            mask = np.logical_and(mask_x, mask_y)
        else:
            _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[0] >= 0, mapping[0] <= w - 1)
            mask_y = np.logical_and(mapping[1] >= 0, mapping[1] <= h - 1)
            mask = np.logical_and(mask_x, mask_y)
        mask = mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") else mask.astype(np.uint8)
    else:
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask = mapping[:, 0].ge(0) & mapping[:, 0].le(w-1) & mapping[:, 1].ge(0) & mapping[:, 1].le(h-1)
        else:
            _, h, w = mapping.shape
            mask = mapping[0].ge(0) & mapping[0].le(w-1) & mapping[1].ge(0) & mapping[1].le(h-1)
        mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
    return mask


def get_mapping_horizontal_flipping(image):
    H, W, C = image.shape
    mapping = np.zeros((H, W, 2), np.float32)
    for j in range(H):
        for i in range(W):
            mapping[j, i, 0] = W - i
            mapping[j, i, 1] = j
    return mapping, remap_using_correspondence_map(image, mapping[:, :, 0], mapping[:, :, 1])


def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(1,2,0).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                mapping[i, :, :, 0] = flow[i, :, :, 0] + X
                mapping[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.transpose(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            mapping[:, :, 0] = flow[:, :, 0] + X
            mapping[:, :, 1] = flow[:, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)


def convert_mapping_to_flow(mapping, output_channel_first=True):
    if not isinstance(mapping, np.ndarray):
        # torch tensor
        if len(mapping.shape) == 4:
            if mapping.shape[1] != 2:
                # size is BxHxWx2
                mapping = mapping.permute(0, 3, 1, 2)

            B, C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if mapping.shape[0] != 2:
                # size is HxWx2
                mapping = mapping.permute(2, 0, 1)

            C, H, W = mapping.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float()  # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()

            flow = mapping - grid  # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(mapping.shape) == 4:
            if mapping.shape[3] != 2:
                # size is Bx2xHxW
                mapping = mapping.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = mapping.shape[:3]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if mapping.shape[0] == 2:
                # size is 2xHxW
                mapping = mapping.transpose(1, 2, 0)
            # HxWx2
            h_scale, w_scale = mapping.shape[:2]
            flow = np.copy(mapping)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:, :, 0] = mapping[:, :, 0] - X
            flow[:, :, 1] = mapping[:, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)


def unormalise_and_convert_mapping_to_flow(map, output_channel_first=True):

    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            B, C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[0, :, :] = (map[0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[1, :, :] = (map[1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if mapping.is_cuda:
                grid = grid.cuda()
            flow = mapping - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        flow = np.copy(map)
        if len(map.shape) == 4:
            if map.shape[1] == 2:
                # size is Bx2xHxWx
                map = map.transpose(0, 2, 3, 1)

            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            mapping = np.zeros_like(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            mapping[:,:,:,0] = (map[:,:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,:,1] = (map[:,:,:,1] + 1) * (h_scale - 1) / 2
            for i in range(b):
                flow[i, :, :, 0] = mapping[i, :, :, 0] - X
                flow[i, :, :, 1] = mapping[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0, 3, 1, 2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.transpose(1, 2, 0)

            # HxWx2
            h_scale, w_scale = map.shape[:2]
            mapping = np.zeros_like(map)
            mapping[:,:,0] = (map[:,:,0] + 1) * (w_scale - 1) / 2
            mapping[:,:,1] = (map[:,:,1] + 1) * (h_scale - 1) / 2
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = mapping[:,:,0]-X
            flow[:,:,1] = mapping[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2, 0, 1)
        return flow.astype(np.float32)


def unormalise_flow_or_mapping(map, output_channel_first=True):

    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            B, C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[:, 0, :, :] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[:, 1, :, :] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            if not output_channel_first:
                mapping = mapping.permute(0, 2, 3, 1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            # channel first, here map is normalised to -1;1
            # we put it back to 0,W-1, then convert it to flow
            C, H, W = map.size()
            mapping = torch.zeros_like(map)
            # mesh grid
            mapping[0, :, :] = (map[0, :, :].float().clone() + 1) * (W - 1) / 2.0  # unormalise
            mapping[1, :, :] = (map[1, :, :].float().clone() + 1) * (H - 1) / 2.0  # unormalise

            if not output_channel_first:
                mapping = mapping.permute(1, 2,0 ).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[1] == 2:
                # size is Bx2xHxWx
                map = map.transpose(0, 2, 3, 1)

            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            mapping = np.zeros_like(map)
            mapping[:, :, :, 0] = (map[:, :, :, 0] + 1) * (w_scale - 1) / 2
            mapping[:, :, :, 1] = (map[:, :, :, 1] + 1) * (h_scale - 1) / 2

            if output_channel_first:
                mapping = mapping.transpose(0, 3, 1, 2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.transpose(1, 2, 0)

            # HxWx2
            h_scale, w_scale = map.shape[:2]
            mapping = np.zeros_like(map)
            mapping[:, :, 0] = (map[:, :, 0] + 1) * (w_scale - 1) / 2
            mapping[:, :, 1] = (map[:, :, 1] + 1) * (h_scale - 1) / 2

            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)


def unnormalize(tensor, output_channel_first=True):
    if len(tensor.shape) == 4:
        if tensor.shape[1] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(0, 3, 1, 2)
            else:
                tensor = tensor.transpose(0, 3, 1, 2)

        B, C, H, W = tensor.size()
        tensor_unnorm = torch.zeros_like(tensor)
        # mesh grid
        tensor_unnorm[:, 0, :, :] = (tensor[:, 0, :, :] + 1) * (W - 1) / 2.0  # unormalise
        tensor_unnorm[:, 1, :, :] = (tensor[:, 1, :, :] + 1) * (H - 1) / 2.0  # unormalise

        if not output_channel_first:
            tensor_unnorm = tensor_unnorm.permute(0, 2, 3, 1)
    else:
        if tensor.shape[0] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor.transpose(2, 0, 1)


        C, H, W = tensor.size()
        tensor_unnorm = torch.zeros_like(tensor)
        # mesh grid
        tensor_unnorm[0, :, :] = (tensor[0, :, :] + 1) * (W - 1) / 2.0  # unormalise
        tensor_unnorm[1, :, :] = (tensor[1, :, :] + 1) * (H - 1) / 2.0  # unormalise

        if not output_channel_first:
            tensor_unnorm = tensor_unnorm.permute(0, 2, 3, 1)

    return tensor_unnorm


def normalize(tensor, output_channel_first=True):
    if len(tensor.shape) == 4:
        if tensor.shape[1] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(0, 3, 1, 2)
            else:
                tensor = tensor.transpose(0, 3, 1, 2)

        B, C, H, W = tensor.size()
        tensor_norm = torch.zeros_like(tensor)
        # mesh grid
        tensor_norm[:, 0, :, :] = 2 * tensor[:, 0, :, :] / (W - 1) - 1.0
        tensor_norm[:, 1, :, :] = 2 * tensor[:, 1, :, :] / (H - 1) - 1.0

        if not output_channel_first:
            tensor_norm = tensor_norm.permute(0, 2, 3, 1)
    else:
        if tensor.shape[0] != 2:
            if not isinstance(map, np.ndarray):
                tensor = tensor.permute(2, 0, 1)
            else:
                tensor = tensor.transpose(2, 0, 1)

        C, H, W = tensor.size()
        tensor_norm = torch.zeros_like(tensor)
        # mesh grid
        tensor_norm[0, :, :] = 2 * tensor[0, :, :] / (W - 1) - 1.0
        tensor_norm[1, :, :] = 2 * tensor[1, :, :] / (H - 1) - 1.0

        if not output_channel_first:
            tensor_norm = tensor_norm.permute(0, 2, 3, 1)

    return tensor_norm
