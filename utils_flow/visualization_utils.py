import re
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array
except:
    print('Did not load moviepy')
import os
import numpy as np
import cv2

from utils_flow.img_processing_utils import pad_to_same_shape


def horizontal_combine_images(img1, img2):
    ratio = img1.shape[0] / img2.shape[0]
    imgs_comb = np.hstack((img1, cv2.resize(img2, None, fx=ratio, fy=ratio)))
    return imgs_comb


def draw_matches(img1, img2, kp1, kp2):
    """

    Args:
        img1:
        img2:
        kp1: kp1 is shape Nx2, N number of feature points, first point in horizontal direction
        kp2: kp2 is shape Nx2, N number of feature points, first point in horizontal direction

    Returns:

    """
    img1, img2 = pad_to_same_shape(img1, img2)
    h, w = img1.shape[:2]
    img = horizontal_combine_images(img1, img2)

    if kp1.shape[0] == 0:
        return img
    # shape Mx1x2 M number of matches
    kp2[:, 0] = kp2[:, 0] + w

    for i in range(kp1.shape[0]):
        img = cv2.line(img, (kp1[i, 0], kp1[i, 1]), (kp2[i, 0], kp2[i, 1]), (255, 0, 0), 2)
    return img


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


def make_sparse_matching_plot(image0, image1, mkpts0, mkpts1, color, margin=10):
    H0, W0 = image0.shape[:2]
    H1, W1 = image1.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)
    return out


def draw_keypoints(img, kp):
    """

    Args:
        img:
        kp: kp1 is shape Nx2, N number of feature points, first point in horizontal direction

    Returns:

    """
    image_copy = np.copy(img)
    nbr_points = kp.shape[0]
    for i in range(nbr_points):
        image = cv2.circle(image_copy, (np.uint(kp[i,0]),np.uint(kp[i,1])), 1, (0,255,0),thickness=5)
    return image


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


def make_and_save_video(image_folder, video_name, rate=8):
    print(image_folder)
    images = sorted_nicely([img for img in os.listdir(image_folder) if (img.endswith(".png") or img.endswith(".jpg"))])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"XVID"), rate, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def put_video_side_by_side(directory, name, model):
    clip_source = VideoFileClip(os.path.join(directory, '{}_source.mp4'.format(name)))
    clip_target = VideoFileClip(os.path.join(directory, '{}_target.mp4'.format(name)))
    clip_model = VideoFileClip(os.path.join(directory, '{}_warped_source_masked.mp4'.format(name)))
    return clips_array([[clip_source, clip_target, clip_model]])


def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def overlay_with_colored_mask(im, mask, alpha=0.5):
    fg = im * alpha + (1 - alpha) * mask
    return fg


def overlay_semantic_mask(im, ann, alpha=0.5, mask=None, colors=None, color=[255, 218, 185], contour_thickness=1):
    """
    example usage:
    image_overlaid = overlay_semantic_mask(im.astype(np.uint8), 255 - mask.astype(np.uint8) * 255, color=[255, 102, 51])
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.int)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)
    colors[-1, :] = color

    if mask is None:
        mask = colors[ann]

    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]  # where the mask is zero (where object is), shoudlnt be any color

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, color,
                             contour_thickness)
    return img


def replace_area(im, ann, replace, alpha=0.5, color=None, thickness=1):
    img_warped_overlay_on_target = np.copy(replace)
    img_warped_overlay_on_target[ann > 0] = im[ann > 0]
    for obj_id in np.unique(ann[ann > 0]):
        contours = cv2.findContours((ann == obj_id).astype(
            np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(img_warped_overlay_on_target, contours[0], -1, color,
                         thickness)
    return img_warped_overlay_on_target