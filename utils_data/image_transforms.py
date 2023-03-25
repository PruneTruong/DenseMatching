import torch
import numpy as np
import torch.nn.functional as F
import random
import math


def TensorToArray(tensor, type):
    """Converts a torch.FloatTensor of shape (C x H x W) to a numpy.ndarray (H x W x C) """
    array=tensor.cpu().detach().numpy()
    if len(array.shape) == 4:
        if array.shape[3] > array.shape[1]:
            # shape is BxCxHxW
            array = np.transpose(array, (0,2,3,1))
    else:
        if array.shape[2] > array.shape[0]:
            # shape is CxHxW
            array=np.transpose(array, (1,2,0))
    return array.astype(type)


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W), where values are in [0, 1]."""

    def __call__(self, img, *args, **kwargs):
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if len(img.shape) == 2:
            img = img[:, :, None]

        img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img)
        # put it from HWC to CHW forma
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            # supposed to be float tensor, within [0, 1]
            assert torch.max(img).le(1.0)
            return img


# class ToTensor of torchvision also normalised to 0 1
class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, get_float=True):
        self.get_float = get_float

    def __call__(self, array):

        if not isinstance(array, np.ndarray):
            array = np.array(array)
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        if self.get_float:
            # carefull, this is not normalized to [0, 1]
            return tensor.float()
        else:
            return tensor


class PILToNumpy(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, get_float=True):
        self.get_float = get_float

    def __call__(self, array):

        array = np.array(array)
        return array


class ScaleToZeroOne(object):
    """Converts torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, array):
        return array.float().div(255)


class RandomColorWarp(object):
    """Applies random color warp to a numpy HxWx3 image"""
    def __init__(self, mean_range=0, std_range=0, gamma=[0.7, 1.2], contrast_change=[0.5, 1.5]):
        self.mean_range = mean_range
        self.std_range = std_range
        self.gamma = gamma
        self.contrast_change = contrast_change

    def __call__(self, image):

        # adjust contrast
        contrast = random.uniform(self.contrast_change[0], self.contrast_change[1])
        mean = np.mean(image, axis=(0, 1))
        image = mean + (image - mean) * contrast
        image = np.clip(image, 0, 255)

        # apply gamma augmentation
        random_gamma = random.uniform(self.gamma[0], self.gamma[1])
        image = (image/255.0) ** random_gamma * 255.0

        # multiplicative brighness changes (per image)
        random_std = random.uniform(-self.std_range, self.std_range)
        image *= (1 + random_std)

        # additive brightness change
        random_mean = random.uniform(-self.mean_range, self.mean_range)
        image += random_mean

        # random_order = np.random.permutation(3)
        # image = image[:,:,random_order]

        image = np.clip(image, 0, 255)
        return image.astype(np.uint8)


class RGBtoBGR(object):
    """converts the RGB channels of a numpy array HxWxC into BGR"""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        ch_arr = [2, 1, 0]
        img = array[..., ch_arr]
        return img


# Flow transform here
class ResizeFlow(object):
    """Resize a provided flow field (must be in shape 2xHxW to the given size."""
    def __init__(self, size):
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size

    def __call__(self, tensor):
        assert(tensor.shape[0] == 2 and len(tensor.shape) == 3)
        _, h_original, w_original = tensor.shape
        resized_tensor = F.interpolate(tensor.unsqueeze(0), self.size, mode='bilinear', align_corners=False)
        resized_tensor[:, 0, :, :] *= float(self.size[1])/float(w_original)
        resized_tensor[:, 1, :, :] *= float(self.size[0])/float(h_original)
        return resized_tensor.squeeze(0)


class Blur(torch.nn.Module):
    """ Blur the image by applying a gaussian kernel with given sigma.
    Image must be tensor or numpy, 3 dimensional. """

    def __init__(self, sigma):
        super().__init__()
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2 * s) for s in self.sigma]
        x_coord = [torch.arange(-sz, sz + 1, dtype=torch.float32) for sz in self.filter_size]
        self.filter = [torch.exp(-(x ** 2) / (2 * s ** 2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1, 1, -1, 1) / self.filter[0].sum()
        self.filter[1] = self.filter[1].view(1, 1, 1, -1) / self.filter[1].sum()

    def forward(self, image):
        if torch.is_tensor(image):
            sz = image.shape[2:]
            im1 = F.conv2d(image.view(-1, 1, sz[0], sz[1]), self.filter[0], padding=(self.filter_size[0], 0))
            return F.conv2d(im1, self.filter[1], padding=(0, self.filter_size[1])).view(-1, sz[0], sz[1])
        else:
            raise NotImplementedError


def to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    This function does not support torchscript.
    See :class:`~torchvision.transforms.ToTensor` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """

    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if len(pic.shape) == 4:
            if pic.shape[1] != 3 or pic.shape[1] != 1:
                pic = pic.transpose((0, 3, 1, 2))
        elif len(pic.shape) == 3:
            if pic.shape[0] != 3 or pic.shape[0] != 1:
                pic = pic.transpose((2, 0, 1))
        if pic.ndim == 2:
            pic = pic[:, :, None]
            pic = pic.transpose((2, 0, 1))

        img = torch.from_numpy(pic).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype).div(255)
        else:
            return img
    else:
        raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))


class RandomBlur(torch.nn.Module):
    """ Blur the image, with a given probability, by applying a gaussian kernel with given sigma.
    Image must be tensor or numpy, 3 dimensional. """

    def __init__(self, sigma=(0.2, 2.0), kernel_size=(3, 7), probability=0.1):
        super().__init__()
        self.probability = probability
        self.sigma = sigma
        self.kernel_size = kernel_size

    def get_params(self, sigma_min: float, sigma_max: float):
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        kernel_size = random.randint(self.kernel_size[0], self.kernel_size[1])
        if kernel_size % 2 == 0:
            kernel_size += 1
        return torch.empty(1).uniform_(sigma_min, sigma_max).item(), kernel_size

    def forward(self, image, do_blur=True):
        # only apply blur with a probability
        if random.random() < self.probability:

            # sample random sigma
            sigma, filter_size = self.get_params(self.sigma[0], self.sigma[1])

            sigma = (sigma, sigma)
            filter_size = [filter_size, filter_size]
            # filter_size = [math.ceil(2 * s) for s in sigma]
            x_coord = [torch.arange(-sz, sz + 1, dtype=torch.float32) for sz in filter_size]
            filter = [torch.exp(-(x ** 2) / (2 * s ** 2)) for x, s in zip(x_coord, sigma)]
            filter[0] = filter[0].view(1, 1, -1, 1) / filter[0].sum()
            filter[1] = filter[1].view(1, 1, 1, -1) / filter[1].sum()

            to_numpy = False
            if isinstance(image, np.ndarray):
                image = to_tensor(image)  # C, H, W and in range [0, 1]
                to_numpy = True

            if image.shape[0] != 3:
                image = image.permute(3, 0, 1)

            if torch.is_tensor(image):
                sz = image.shape[1:]
                im1 = F.conv2d(image.view(-1, 1, sz[0], sz[1]), filter[0].to(image.device), padding=(filter_size[0], 0))
                img_blur = F.conv2d(im1, filter[1].to(image.device), padding=(0, filter_size[1])).view(-1, sz[0], sz[1])

                if to_numpy:
                    if img_blur.is_floating_point():
                        img_blur = img_blur.mul(255).byte()
                    img_blur = np.transpose(img_blur.cpu().numpy(), (1, 2, 0))
                return img_blur
            else:
                raise NotImplementedError
        else:
            return image