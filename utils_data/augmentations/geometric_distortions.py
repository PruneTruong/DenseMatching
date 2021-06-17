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


class RadialDistortion:
    def __init__(self, parameters):
        self.shift_limit = to_tuple(parameters["shift_limit"])
        self.k1_limit= (parameters["min_k1"], parameters["max_k1"])
        self.k2_limit= (parameters["min_k2"], parameters["max_k2"])
        self.p1_limit= (parameters["min_p1"], parameters["max_p1"])
        self.p2_limit= (parameters["min_p2"], parameters["max_p2"])

    def get_random_parameters(self):
        k1= random.uniform(self.k1_limit[0], self.k1_limit[1])
        k2= random.uniform(self.k2_limit[0], self.k2_limit[1])
        p1= random.uniform(self.p1_limit[0], self.p1_limit[1])
        p2= random.uniform(self.p2_limit[0], self.p2_limit[1])
        dx=round(random.uniform(self.shift_limit[0], self.shift_limit[1]))
        dy=round(random.uniform(self.shift_limit[0], self.shift_limit[1]))
        return np.array([k1, k2, p1, p2]), (dx, dy)

    def get_mapping_from_distorted_image_to_undistorted(self, shape, distCoeffs, shift=(0,0)):
        return get_mapping_from_radially_distorted_image_to_undistorted(shape, distCoeffs, shift)

    def get_mapping_from_undistorted_to_distorted_image(self, shape, distCoeffs, shift=(0,0)):
        return get_mapping_from_undistorted_to_radially_distorted_image(shape,  distCoeffs, shift)


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
    def __init__(self, parameters, get_flow=False, approximate=False):
        self.sigma_params=(parameters["min_sigma"], parameters["max_sigma"])
        self.alpha_params=(parameters["min_alpha"], parameters["max_alpha"])
        self.get_flow = get_flow
        self.approximate = approximate

    def get_random_paremeters(self, shape, seed=None):
        #sigma_params=(0.05, 0.05), alpha_params=(1, 5)
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

    def get_mapping_from_undistorted_image_to_distorted(self, shape, sigma, alpha, seed=None):
        # i think this is false
        return inverse_elastic_transform(shape, sigma, alpha, seed, get_flow=self.get_flow)


def get_mapping_from_radially_distorted_image_to_undistorted(shape,  distCoeffs, shift=(0,0)):
    '''
    for every pixel X and Y in distorted image, calculates corresponding pixel map_x and map_y in undistorted image
    # according to cv2.undistort_points
    :param shape:
    :param K:
    :param distCoeffs:
    :return:
    '''
    (h_scale, w_scale)=shape[:2]
    fx = w_scale
    fy = h_scale

    cx = w_scale * 0.5 + shift[0]
    cy = h_scale * 0.5 + shift[1]

    K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    N=len(X.flatten())
    distorted_points=np.concatenate((X.flatten().reshape(N,1), Y.flatten().reshape(N,1)), axis=1)
    undistorted_points=cv2.undistortPoints(distorted_points.reshape(N,1,2), K, distCoeffs, R=None, P=K)

    map_x=undistorted_points[:,0,0].reshape((h_scale,w_scale))
    map_y=undistorted_points[:,0,1].reshape((h_scale,w_scale))
    return map_x.astype(np.float32), map_y.astype(np.float32)


def get_mapping_from_undistorted_to_radially_distorted_image(shape,distCoeffs, shift=(0,0)):
    '''
    for every pixel X,Y in the original image (not distorted), calculates corresponding index in distorted image
    according to inverse mapping used in cv2.initUndistortRectifyMap
    :param shape:
    :param K:
    :param distCoeffs:
    :return:
    '''
    (h_scale, w_scale)=shape[:2]
    fx = w_scale
    fy = h_scale

    cx = w_scale * 0.5 + shift[0]
    cy = h_scale * 0.5 + shift[1]

    K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                       np.linspace(0, h_scale - 1, h_scale))
    x_camera=(X-K[0,2])/K[0,0]
    y_camera=(Y-K[1,2])/K[1,1]
    r_square=x_camera**2 + y_camera**2
    x_camera_prime=x_camera*(1+distCoeffs[0]*r_square+ distCoeffs[1]*r_square**2) + \
                   2*distCoeffs[2]*x_camera*y_camera + \
                   distCoeffs[3]*(r_square + 2*x_camera**2)
    y_camera_prime=y_camera*(1+distCoeffs[0]*r_square+ distCoeffs[1]*r_square**2) + \
                   2*distCoeffs[3]*x_camera*y_camera + \
                   distCoeffs[2]*(r_square+2*y_camera**2)
    map_x=x_camera_prime*K[0,0]+K[0,2]
    map_y=y_camera_prime*K[1,1]+K[1,2]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def get_mapping_from_undistorted_to_distorted_image_2(shape, distCoeffs, shift=(0,0)):
    """Correction of Barrel / pincushion distortion. Unconventional augment.
    Reference:
        |  https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        |  https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        |  https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
        |  http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/
    """
    (h_scale, w_scale)=shape[:2]
    fx = w_scale
    fy = h_scale

    cx = w_scale * 0.5 + shift[0]
    cy = h_scale * 0.5 + shift[1]

    K = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    map_x, map_y = cv2.initUndistortRectifyMap(K, distCoeffs, None, K, (w_scale, h_scale), cv2.CV_32FC1)
    return map_x, map_y


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


def inverse_elastic_transform(shape, sigma, alpha, seed=None, padding=10, get_flow=False ):
    """ Apply an elastic distortion to the image
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
    shape=shape[:2]

    # Create the grid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    inverse_map_x = np.float32(x - dx)
    inverse_map_y = np.float32(y - dy)
    return inverse_map_x, inverse_map_y


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

