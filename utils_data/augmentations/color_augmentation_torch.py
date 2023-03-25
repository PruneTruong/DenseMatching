import numbers
import numpy as np
import torch
from packaging import version
import random
from collections.abc import Sequence
from torch import Tensor
from torch.nn.functional import grid_sample, conv2d, interpolate, pad as torch_pad


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


def rgb_to_grayscale(img, num_output_channels=1):
    if len(img.shape) < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(len(img.shape)))

    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')

    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img


def _rgb2hsv(img):
    r, g, b = img.unbind(dim=-3)

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    if version.parse(torch.__version__) >= version.parse("1.5"):
        maxc = torch.max(img, dim=-3).values
        minc = torch.min(img, dim=-3).values
    else:
        maxc = torch.max(img, dim=-3)
        minc = torch.min(img, dim=-3)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = torch.eq(maxc, r).float() * (bc - gc)
    hg = (torch.eq(maxc, g) & ~torch.eq(maxc, r)).float() * (2.0 + rc - bc)
    hb = (~torch.eq(maxc, g) & ~torch.eq(maxc, r)).float() * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-3)


def _hsv2rgb(img):
    h, s, v = img.unbind(dim=-3)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = torch.eq(i.unsqueeze(dim=-3), torch.arange(6, device=i.device, dtype=i.dtype).view(-1, 1, 1))

    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    return torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=img.dtype), a4)


class ColorJitter(torch.nn.Module):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, invert_channel=True):
        """Randomly change the brightness, contrast and saturation of an image.

        Args:
            brightness (float or tuple of float (min, max)): How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            contrast (float or tuple of float (min, max)): How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
            saturation (float or tuple of float (min, max)): How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            hue (float or tuple of float (min, max)): How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            invert_channel (bool): creates more drastic transformations by inverting order of RGB channels.

        """
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.invert_channel = invert_channel

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def _blend(self, img1, img2, ratio):
        ratio = float(ratio)
        bound = 1.0 if img1.is_floating_point() else 255.0
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)

    def adjust_brightness(self, img,  brightness_factor):
        return self._blend(img, torch.zeros_like(img), brightness_factor)

    def adjust_contrast(self, img, contrast_factor):
        dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        mean = torch.mean(rgb_to_grayscale(img).to(dtype), dim=(-3, -2, -1), keepdim=True)
        return self._blend(img, mean, contrast_factor)

    def adjust_hue(self, img, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

        if not (isinstance(img, torch.Tensor)):
            raise TypeError('Input img should be Tensor image')

        orig_dtype = img.dtype
        if img.dtype == torch.uint8:
            img = img.to(dtype=torch.float32) / 255.0

        img = _rgb2hsv(img)
        h, s, v = img.unbind(dim=-3)
        h = (h + hue_factor) % 1.0
        img = torch.stack((h, s, v), dim=-3)
        img_hue_adj = _hsv2rgb(img)

        if orig_dtype == torch.uint8:
            img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

        return img_hue_adj

    def adjust_saturation(self, img, saturation_factor):
        if saturation_factor < 0:
            raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))
        return self._blend(img, rgb_to_grayscale(img), saturation_factor)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        orig_dtype = img.dtype
        if orig_dtype == torch.uint8:
            img = img.float()
            img /= 255.0
        fn_idx = torch.randperm(3)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = self.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = self.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = self.adjust_saturation(img, saturation_factor)

            '''
            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = self.adjust_hue(img, hue_factor)
            '''

        if orig_dtype == torch.uint8:
            img *= 255.0
            img = torch.clamp(img, 0, max=255)
            img = img.byte()

        if self.invert_channel:
            random_order = np.random.permutation(3)
            img = img[:, random_order]
        return img


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _get_gaussian_kernel1d(kernel_size: int, sigma: float):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
        kernel_size, sigma, dtype: torch.dtype, device: torch.device
):
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def _cast_squeeze_in(img: Tensor, req_dtypes):
    need_squeeze = False
    # make image NCHW
    if len(img.shape) < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)
    return img, need_cast, need_squeeze, out_dtype


def _cast_squeeze_out(img: Tensor, need_cast: bool, need_squeeze: bool, out_dtype: torch.dtype):
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # it is better to round before cast
            img = torch.round(img)
        img = img.to(out_dtype)

    return img


class RandomGaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    The image can be a numpy Image or a Tensor.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        Numpy Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size=(3, 7), sigma=(0.1, 2.0), probability=0.1):
        super().__init__()
        self.probability = probability
        self.kernel_size = kernel_size

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")
        self.sigma = sigma

    def gaussian_blur(self, img, kernel_size, sigma):
        if not (isinstance(img, torch.Tensor)):
            raise TypeError('img should be Tensor. Got {}'.format(type(img)))

        dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=dtype, device=img.device)
        kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])

        img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype, ])

        # padding = (left, right, top, bottom)
        padding = [kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[1] // 2]
        img = torch_pad(img, padding, mode="reflect")
        img = conv2d(img, kernel, groups=img.shape[-3])
        img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
        return img

    def get_params(self, sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            sigma: Standard deviation to be passed to calculate kernel for gaussian blurring.
            kernel_size: Kernel size to be passed for gaussian blurring
        """
        sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()

        kernel_size = random.randint(self.kernel_size[0], self.kernel_size[1])
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        return sigma, kernel_size

    def blur_4d_tensor(self, img):
        sigma, kernel_size = self.get_params(self.sigma[0], self.sigma[1])
        img_blur = self.gaussian_blur(img, kernel_size, [sigma, sigma])
        return img_blur

    def forward(self, img):
        """
        Args:
            img (numpy or Tensor): image to be blurred.

        Returns:
            numpy or Tensor: Gaussian blurred image
        """
        if random.random() < self.probability:
            to_numpy = False
            if isinstance(img, np.ndarray):
                img = to_tensor(img)
                to_numpy = True

            if len(img.shape) == 3:
                img = img.unsqueeze(0)
                img_blur = self.blur_4d_tensor(img).squeeze(0)
            else:
                img_blur = self.blur_4d_tensor(img)

            if to_numpy:
                if img_blur.is_floating_point():
                    img_blur = img_blur.mul(255).byte()
                img_blur = np.transpose(img_blur.cpu().numpy(), (1, 2, 0))
            return img_blur
        else:
            return img

    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s