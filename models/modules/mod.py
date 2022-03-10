import torch
import torch.nn as nn
import numpy as np


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, batch_norm=False, relu=True):
    if batch_norm:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.LeakyReLU(0.1, inplace=True))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, bias=bias),
                                nn.BatchNorm2d(out_planes))
    else:
        if relu:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias),
                                nn.LeakyReLU(0.1))
        else:
            return nn.Sequential(
                                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation, bias=bias))


def predict_flow(in_planes, nbr_out_channels=2):
    return nn.Conv2d(in_planes, nbr_out_channels, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    deconv_ = nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

    nn.init.kaiming_normal_(deconv_.weight.data, mode='fan_in')
    if deconv_.bias is not None:
        deconv_.bias.data.zero_()
    return deconv_


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
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
    flow = mapping - grid
    return flow


class OpticalFlowEstimator(nn.Module):
    """
    Original PWCNet optical flow decoder. With DenseNet connections.
    """
    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimator, self).__init__()

        dd = np.cumsum([128,128,96,64,32])
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(in_channels + dd[0], 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(in_channels + dd[1], 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(in_channels + dd[2], 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(in_channels + dd[3], 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(in_channels + dd[4])

    def forward(self, x):
        # dense net connection
        x = torch.cat((self.conv_0(x), x),1)
        x = torch.cat((self.conv_1(x), x),1)
        x = torch.cat((self.conv_2(x), x),1)
        x = torch.cat((self.conv_3(x), x),1)
        x = torch.cat((self.conv_4(x), x),1)
        flow = self.predict_flow(x)
        return x, flow


class OpticalFlowEstimatorNoDenseConnection(nn.Module):
    """
    PWCNet optical flow decoder modified with feed forward connections.
    """
    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimatorNoDenseConnection, self).__init__()

        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.predict_flow = predict_flow(32)

    def forward(self, x):
        x = self.conv_4(self.conv_3(self.conv_2(self.conv_1(self.conv_0(x)))))
        flow = self.predict_flow(x)
        return x, flow


class OpticalFlowEstimatorResidualConnection(nn.Module):
    """
    PWCNet optical flow decoder modified with residual connections.
    """
    def __init__(self, in_channels, batch_norm):
        super(OpticalFlowEstimatorResidualConnection, self).__init__()

        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv0_skip = conv(128, 96, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv2_skip = conv(96, 32, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.predict_flow = predict_flow(32)

    def forward(self, x):
        x0 = self.conv_0(x)
        x0_relu = self.leakyRELU(x0)
        x2 = self.conv_2(self.conv_1(x0_relu))
        x2_skip = x2 + self.conv0_skip(x0)
        x2_skip_relu = self.leakyRELU(x2_skip)
        x4 = self.conv_4(self.conv_3(x2_skip_relu))
        x4_skip = x4 + self.conv2_skip(x2_skip)
        x = self.leakyRELU(x4_skip)
        flow = self.predict_flow(x)

        return x, flow


# extracted from DGCNet
def conv_blck(in_channels, out_channels, kernel_size=3,
              stride=1, padding=1, dilation=1, bn=False):
    if bn:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation),
                             nn.ReLU(inplace=True))


def conv_head(in_channels, nbr_out_channels=2):
    return nn.Conv2d(in_channels, nbr_out_channels, kernel_size=3, padding=1)


class CorrespondenceMapBase(nn.Module):
    def __init__(self, in_channels, bn=False):
        super().__init__()

    def forward(self, x1, x2=None, x3=None):
        x = x1
        # concatenating dimensions
        if (x2 is not None) and (x3 is None):
            x = torch.cat((x1, x2), 1)
        elif (x2 is None) and (x3 is not None):
            x = torch.cat((x1, x3), 1)
        elif (x2 is not None) and (x3 is not None):
            x = torch.cat((x1, x2, x3), 1)

        return x


class CMDTop(CorrespondenceMapBase):
    """
    original DGC-Net mapping decoder.
    """
    def __init__(self, in_channels, batch_norm=False, output_x=False):
        super().__init__(in_channels, batch_norm)

        self.output_x = output_x
        chan = [128, 128, 96, 64, 32]
        self.conv0 = conv_blck(in_channels, chan[0], bn=batch_norm)
        self.conv1 = conv_blck(chan[0], chan[1], bn=batch_norm)
        self.conv2 = conv_blck(chan[1], chan[2], bn=batch_norm)
        self.conv3 = conv_blck(chan[2], chan[3], bn=batch_norm)
        self.conv4 = conv_blck(chan[3], chan[4], bn=batch_norm)
        self.final = conv_head(chan[-1])

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x = self.conv4(self.conv3(self.conv2(self.conv1(self.conv0(x)))))
        mapping = self.final(x)
        if self.output_x:
            return x, mapping
        else:
            return mapping


class CMDTopResidualConnections(CorrespondenceMapBase):
    """
    DGC-Net mapping decoder, with residual connections.
    """
    def __init__(self, in_channels, batch_norm=False, output_x=False):
        super().__init__(in_channels, batch_norm)

        self.output_x = output_x
        self.conv_0 = conv(in_channels, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv0_skip = conv(128, 96, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_1 = conv(128, 128, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_2 = conv(128, 96, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.conv2_skip = conv(96, 32, kernel_size=1, stride=1, padding=0, batch_norm=batch_norm, relu=False, bias=False)
        self.conv_3 = conv(96, 64, kernel_size=3, stride=1, batch_norm=batch_norm, relu=True)
        self.conv_4 = conv(64, 32, kernel_size=3, stride=1, batch_norm=batch_norm, relu=False)
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.final = conv_head(32)

    def forward(self, x1, x2=None, x3=None):
        x = super().forward(x1, x2, x3)
        x0 = self.conv_0(x)
        x0_relu = self.leakyRELU(x0)
        x2 = self.conv_2(self.conv_1(x0_relu))
        x2_skip = x2 + self.conv0_skip(x0)
        x2_skip_relu = self.leakyRELU(x2_skip)
        x4 = self.conv_4(self.conv_3(x2_skip_relu))
        x4_skip = x4 + self.conv2_skip(x2_skip)
        x = self.leakyRELU(x4_skip)

        mapping = self.final(x)
        if self.output_x:
            return x, mapping
        else:
            return mapping
