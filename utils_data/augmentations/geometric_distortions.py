import cv2
import numpy as np
from functools import wraps
from scipy.ndimage.filters import gaussian_filter
import random


def preserve_shape(func):
    """Preserve shape of the image."""
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


class GridDistortion:
    """
    https://github.com/albu/albumentations
    Targets:
        image, mask
    Image types:
        uint8, float32
    """
    def __init__(self, parameters):
        self.num_steps = parameters["num_steps"]
        self.distort_limit = to_tuple(parameters["distort_limit"])

    def get_mapping_from_distorted_image_to_undistorted(self, shape, stepsx=[], stepsy=[]):
        return grid_distortion(shape, self.num_steps, stepsx, stepsy)

    def get_random_parameters(self):
        # num_steps=5, distort_limit=0.3
        stepsx = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        stepsy = [1 + random.uniform(self.distort_limit[0], self.distort_limit[1]) for i in
                  range(self.num_steps + 1)]
        return stepsx, stepsy


class ElasticTransform:
    def __init__(self, parameters, get_flow=False, approximate=True):
        self.sigma_params = (parameters["min_sigma"], parameters["max_sigma"])
        self.alpha_params = (parameters["min_alpha"], parameters["max_alpha"])
        self.get_flow = get_flow
        self.approximate = approximate

    def get_random_paremeters(self, shape, seed=None):
        # sigma_params=(0.05, 0.05), alpha_params=(1, 5)
        if seed is None:
            random_state = np.random.RandomState(seed)
        else:
            random_state = np.random.RandomState(None)
        shape = shape[:2]
        sigma = np.max(shape) * (self.sigma_params[0] + self.sigma_params[1] * random_state.rand())
        alpha = np.max(shape) * (self.alpha_params[0] + self.alpha_params[1] * random_state.rand())
        return sigma, alpha

    def get_mapping_from_distorted_image_to_undistorted(self, shape, sigma, alpha, seed=None):
        return elastic_transform(shape, sigma, alpha, seed, get_flow=self.get_flow, approximate=self.approximate)


def grid_distortion(shape, num_steps=10, xsteps=[], ysteps=[]):
    """
    Reference:
        https://github.com/albu/albumentations
        http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
    """
    height, width = shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    return map_x.reshape((height,width)), map_y.reshape((height,width))


def elastic_transform(shape, sigma, alpha, seed=None, padding=10, get_flow=False, approximate=False):
    """ Apply an elastic distortion to the image
    https://github.com/albu/albumentations
    Parameters:
      sigma_params: sigma can vary between max(img.shape) * sigma_params[0] and
                    max(img.shape) * (sigma_params[0] + sigma_params[1])
      alpha_params: alpha can vary between max(img.shape) * alpha_params[0] and
                    max(img.shape) * (alpha_params[0] + alpha_params[1])
      padding: padding that will be removed when cropping (remove strange artefacts)
    """
    if seed is None:
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(None)
    shape = shape[:2]
    [height, width] = shape

    # Create the grid
    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (0,0), sigma, dst=dx)
        dx *= alpha

        dy = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dy, (0,0), sigma, dst=dy)
        dy *= alpha
    else:
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    if get_flow:
        return dx.astype(np.float32), dy.astype(np.float32)

    else:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # Apply the distortion
        map_x=np.float32(x + dx)
        map_y=np.float32(y + dy)
        return map_x, map_y


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple
    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError('Arguments low and bias are mutually exclusive')

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = - param, + param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError('Argument param must be either scalar (int,float) or tuple')

    if bias is not None:
        return tuple([bias + x for x in param])

    return tuple(param)

