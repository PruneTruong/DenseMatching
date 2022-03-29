import torch.utils.data as data
import os
import os.path
import numpy as np
import cv2
import pandas as pd


def default_loader(root, path_imgs):
    image1 = cv2.imread(os.path.join(root, path_imgs[0]))
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)  # it is black and white, convert to BGR
    else:
        image1 = image1[:, :, ::-1] # convert to RGB from BGR

    image2 = cv2.imread(os.path.join(root, path_imgs[1]))
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)  # it is black and white, convert to BGR
    else:
        image2 = image2[:, :, ::-1] # convert to RGB from BGR
    return image1.astype(np.uint8), image2.astype(np.uint8)


def make_dataset(dir):
    '''Will search for pairs of images in the order in which they appear  '''
    images = []
    name_images = [f for f in sorted(os.listdir(dir))]
    i=0
    while i < len(name_images)-1:
        img1 = name_images[i]
        img2 = name_images[i+1]
        images.append([img1, img2])
        i += 2
    return images


class DatasetNoGT(data.Dataset):
    """
    From an image sequence, retrieves pair of images (all from an image folder), by using always the same one as the
    reference/target (usually middle frame), and using all images as the query/source.
    """

    def __init__(self, root, path_csv=None, source_image_transform=None, target_image_transform=None,
                 loader=default_loader, middle_image=None, start=0, end=None):
        """

        Args:
            root: root directory
            path_csv: optional, contains list of images to read
            source_image_transform: image transformations to apply to source images
            target_image_transform: image transformations to apply to target images
            loader: image loader
            middle_image: position in the image sequence of the image used as reference/target
            start: position of the start
            end: position of the end
        """

        self.root = root
        if isinstance(path_csv, str):
            # it is a string, must be read from csv
            self.path_list = pd.read_csv(path_csv)
            self.csv = True
        elif isinstance(path_csv, list):
            # a list is directly given
            self.path_list = path_csv
            self.csv = False
        elif path_csv is None:
            self.path_list = [f for f in sorted(os.listdir(root)) if f.endswith('.png') or f.endswith('.jpg')]
            self.csv = False
        self.first_image_transform = source_image_transform
        self.second_image_transform = target_image_transform
        self.loader = loader

        self.start = start

        if end is None:
            self.end = len(self.path_list)
        else:
            self.end = end
        if middle_image is None:
            self.middle_image = (self.end - self.start) // 2 + start
        else:
            self.middle_image = middle_image

    def __getitem__(self, index):

        im1 = cv2.imread(os.path.join(self.root, self.path_list[self.start + index]), 1).astype(np.uint8)[:, :, ::-1]
        im2 = cv2.imread(os.path.join(self.root, self.path_list[self.middle_image]), 1).astype(np.uint8)[:, :, ::-1]

        im1_shape = im1.shape[:2]
        im2_shape = im2.shape[:2]
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

        if self.first_image_transform is not None:
            im1 = self.first_image_transform(im1)
        if self.second_image_transform is not None:
            im2 = self.second_image_transform(im2)

        return {'source_size': im1_shape,
                'target_size': im2_shape,
                'source_image': im1,
                'target_image': im2
                }

    def __len__(self):
        return self.end - self.start
