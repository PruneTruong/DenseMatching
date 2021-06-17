import numpy as np
import cv2


def resize_keeping_aspect_ratio(image, size):
    h, w, _ = image.shape
    if h > w:
        ratio = float(size) / float(h)
    else:
        ratio = float(size) / float(w)
    new_h = int(h*ratio)
    new_w = int(w*ratio)
    return cv2.resize(image, (new_w, new_h)), ratio


def pad_to_same_shape(im1, im2):
    # pad to same shape
    if im1.shape[0] <= im2.shape[0]:
        pad_y_1 = im2.shape[0] - im1.shape[0]
        pad_y_2 = 0
    else:
        pad_y_1 = 0
        pad_y_2 = im1.shape[0] - im2.shape[0]
    if im1.shape[1] <= im2.shape[1]:
        pad_x_1 = im2.shape[1] - im1.shape[1]
        pad_x_2 = 0
    else:
        pad_x_1 = 0
        pad_x_2 = im1.shape[1] - im2.shape[1]
    im1 = cv2.copyMakeBorder(im1, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    im2 = cv2.copyMakeBorder(im2, 0, pad_y_2, 0, pad_x_2, cv2.BORDER_CONSTANT)
    shape = im1.shape
    return im1, im2


def pad_to_size(im, size):
    # load_size first h then w
    if not isinstance(size, tuple):
        size = (size, size)
    # pad to same shape
    if im.shape[0] < size[0]:
        pad_y_1 = size[0] - im.shape[0]
    else:
        pad_y_1 = 0
    if im.shape[1] < size[1]:
        pad_x_1 = size[1] - im.shape[1]
    else:
        pad_x_1 = 0

    im = cv2.copyMakeBorder(im, 0, pad_y_1, 0, pad_x_1, cv2.BORDER_CONSTANT)
    return im


def center_pad(im, size):
    # load_size first h then w
    if not isinstance(size, tuple):
        size = (size, size)
    # pad to same shape
    if im.shape[0] < size[0]:
        pad_y_1 = size[0] - im.shape[0]
    else:
        pad_y_1 = 0
    if im.shape[1] < size[1]:
        pad_x_1 = size[1] - im.shape[1]
    else:
        pad_x_1 = 0

    im = cv2.copyMakeBorder(im, pad_y_1//2, pad_y_1-pad_y_1//2, pad_x_1//2, pad_x_1-pad_x_1//2, cv2.BORDER_CONSTANT)
    return im


def center_crop(img, size):
    """
    Get the center crop of the input image
    Args:
        img: input image [HxWxC]
        size: load_size of the center crop (tuple) (width, height)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)
        #load_size is W,H

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.int(np.ceil((size[0] - w) / 2))
    if h < size[1]:
        pad_h = np.int(np.ceil((size[1] - h) / 2))
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    h, w = img_pad.shape[:2]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1


def crop(img, size, x1, y1):
    """
    Get the center crop of the input image
    Args:
        img: input image [HxWxC]
        size: load_size of the center crop (tuple) (width, height)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)
        #load_size is W,H

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < (x1 + size[0]):
        pad_w = np.int(np.ceil(((size[0] + x1) - w) / 2))
    if h < (y1+size[1]):
        pad_h = np.int(np.ceil(((y1+size[1]) - h) / 2))
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    h, w = img_pad.shape[:2]
    img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]

    return img_pad, x1, y1
