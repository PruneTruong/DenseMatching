"""Defines the BilinearConvTranspose2d class.
https://gist.github.com/mjstevens777/9d6771c45f444843f9e3dce6a401b183
https://github.com/vlfeat/matconvnet-fcn/blob/master/bilinear_u.m
"""
import torch
import torch.nn as nn


class BilinearConvTranspose2d(nn.ConvTranspose2d):
    """A conv transpose initialized to bilinear interpolation."""

    def __init__(self, in_planes, out_planes, kernel_size=4, stride=2, padding=1, groups=1):
        """Set up the layer.
        Parameters
        ----------
        channels: int
            The number of input and output channels
        stride: int or tuple
            The amount of upsampling to do
        groups: int
            Set to 1 for a standard convolution. Set equal to channels to
            make sure there is no cross-talk between channels.
        """
        if isinstance(stride, int):
            stride = (stride, stride)
        super().__init__(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the weight and bias."""
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.weight, 0)
        bilinear_kernel = self.bilinear_kernel(self.kernel_size)

        # for each output channel, applied bilinear_kernel on the same input channel, and 0 on the other input channels.
        for i in range(self.in_channels):
            if self.groups == 1:
                j = i
            else:
                j = 0
            self.weight.data[i, j] = bilinear_kernel

    @staticmethod
    def bilinear_kernel(kernel_size):
        """Generate a bilinear upsampling kernel."""
        num_dims = len(kernel_size)

        shape = (1,) * num_dims
        bilinear_kernel = torch.ones(*shape)

        # The bilinear kernel is separable in its spatial dimensions
        # Build up the kernel dim by dim
        for dim in range(num_dims):
            kernel = kernel_size[dim]
            factor = (kernel + 1) // 2
            if kernel % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            delta = torch.arange(0, kernel) - center
            channel_filter = (1 - torch.abs(delta / factor))

            # Apply the dim filter to the current dim
            shape = [1] * num_dims
            shape[dim] = kernel
            bilinear_kernel = bilinear_kernel * channel_filter.view(shape)
        # if kenel_size is (4,4), bilinear kernel is (4,4)
        # channel_filter is [0.25, 0.75, 0.75, 0.25]
        return bilinear_kernel

