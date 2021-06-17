import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import transforms


class BasePWCNet(nn.Module):
    """
    Common to all PWC-Net-based architectures
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        if float(torch.__version__[:3]) >= 1.3:
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        else:
            output = nn.functional.grid_sample(x, vgrid)

        # the mask makes a difference here
        mask = torch.ones(x.size()).cuda()
        if float(torch.__version__[:3]) >= 1.3:
            mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
        else:
            mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output * mask

    def pre_process_data(self, source_img, target_img, device):
        b, _, h_scale, w_scale = target_img.shape
        int_preprocessed_width = int(math.floor(math.ceil(w_scale / 64.0) * 64.0))
        int_preprocessed_height = int(math.floor(math.ceil(h_scale / 64.0) * 64.0))

        '''
        source_img = torch.nn.functional.interpolate(input=source_img.float().to(device),
                                                     size=(int_preprocessed_height, int_preprocessed_width),
                                                     mode='area').byte()
        target_img = torch.nn.functional.interpolate(input=target_img.float().to(device),
                                                     size=(int_preprocessed_height, int_preprocessed_width),
                                                     mode='area').byte()
        source_img = source_img.float().div(255.0)
        target_img = target_img.float().div(255.0)
        '''
        # this gives slightly better values
        source_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        target_img_copy = torch.zeros((b, 3, int_preprocessed_height, int_preprocessed_width))
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((int_preprocessed_height, int_preprocessed_width),
                                                          interpolation=2),
                                        transforms.ToTensor()])
        # only /255 the tensor
        for i in range(source_img.shape[0]):
            source_img_copy[i] = transform(source_img[i].byte())
            target_img_copy[i] = transform(target_img[i].byte())

        source_img = source_img_copy
        target_img = target_img_copy

        ratio_x = float(w_scale) / float(int_preprocessed_width)
        ratio_y = float(h_scale) / float(int_preprocessed_height)

        # convert to BGR
        return source_img[:, [2, 1, 0]].to(device), target_img[:, [2, 1, 0]].to(device), ratio_x, ratio_y

    def estimate_flow(self, source_img, target_img, output_shape=None, scaling=1.0, mode='channel_first'):
        """
        Estimates the flow field relating the target to the source image. Returned flow has output_shape if provided,
        otherwise the same dimension than the target image. If scaling is provided, the output shape is the
        target image dimension multiplied by this scaling factor.
        Args:
            source_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            target_img: torch tensor, bx3xHxW in range [0, 255], not normalized yet
            output_shape: int or list of int, or None, output shape of the returned flow field
            scaling: float, scaling factor applied to target image shape, to obtain the outputted flow field dimensions
                     if output_shape is None
            mode: if channel_first, flow has shape b, 2, H, W. Else shape is b, H, W, 2

        Returns:
            flow_est: estimated flow field relating the target to the reference image,resized and scaled to output_shape
                      (can be defined by scaling parameter)
        """
        w_scale = target_img.shape[3]
        h_scale = target_img.shape[2]
        # define output_shape
        if output_shape is None and scaling != 1.0:
            output_shape = (int(h_scale*scaling), int(w_scale*scaling))

        source_img, target_img, ratio_x, ratio_y = self.pre_process_data(source_img, target_img, self.device)
        output = self.forward(target_img, source_img)

        flow_est_list = output['flow_estimates']
        flow_est = self.div * flow_est_list[-1]

        if output_shape is not None:
            flow_est = torch.nn.functional.interpolate(input=flow_est, size=(h_scale, w_scale), mode='bilinear',
                                                       align_corners=False)
            ratio_x *= float(output_shape[1]) / w_scale
            ratio_y *= float(output_shape[0]) / h_scale
        else:
            flow_est = torch.nn.functional.interpolate(input=flow_est, size=(h_scale, w_scale), mode='bilinear',
                                                       align_corners=False)

        flow_est[:, 0, :, :] *= ratio_x
        flow_est[:, 1, :, :] *= ratio_y
        if mode == 'channel_first':
            return flow_est
        else:
            return flow_est.permute(0, 2, 3, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, batch_norm=False):
    if batch_norm:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dilation=dilation, bias=True),
                            nn.BatchNorm2d(out_planes),
                            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
                            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation, bias=True),
                            nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

