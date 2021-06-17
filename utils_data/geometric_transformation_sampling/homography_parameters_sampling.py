import numpy as np
import cv2 as cv
import random


class RandomHomography:
    """Generates a random homography transformation."""
    def __init__(self, p_flip=0.0, max_rotation=0.0, max_shear=0.0, max_scale=0.0, max_ar_factor=0.0,
                 min_perspective=0.0, max_perspective=0.0, max_translation=0.0, pad_amount=0):
        super().__init__()
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_scale = max_scale
        self.max_ar_factor = max_ar_factor
        self.min_perspective = min_perspective
        self.max_perspective = max_perspective
        self.max_translation = max_translation
        self.pad_amount = pad_amount

    def roll(self):
        """Randomly selects homography parameters. """
        do_flip = random.random() < self.p_flip
        theta = random.uniform(-self.max_rotation, self.max_rotation)

        shear_x = random.uniform(-self.max_shear, self.max_shear)
        shear_y = random.uniform(-self.max_shear, self.max_shear)

        ar_factor = np.exp(random.uniform(-self.max_ar_factor, self.max_ar_factor))
        scale_factor = np.exp(random.uniform(-self.max_scale, self.max_scale))

        perspective_x = random.uniform(self.min_perspective, self.max_perspective)
        perspective_y = random.uniform(self.min_perspective, self.max_perspective)

        translation_x = random.uniform(-self.max_translation, self.max_translation)
        translation_y = random.uniform(-self.max_translation, self.max_translation)

        return do_flip, theta, (shear_x, shear_y), (scale_factor, scale_factor * ar_factor), \
               (perspective_x, perspective_y), translation_x, translation_y

    def _construct_t_mat(self, image_shape, do_flip, theta, shear_values, scale_factors, tx, ty, perspective_factor):
        """Constructs random homography transform. Usually after calling self.roll() to generate
        the random parameters. """
        im_h, im_w = image_shape
        t_mat = np.identity(3)

        if do_flip:
            t_mat[0, 0] = -1.0
            t_mat[0, 2] = im_w

        t_rot = cv.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
        t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

        t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                            [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                            [0.0, 0.0, 1.0]])

        t_scale = np.array([[scale_factors[0], 0.0, (1.0 - scale_factors[0]) * 0.25 * im_w],
                            [0.0, scale_factors[1], (1.0 - scale_factors[1]) * 0.25 * im_h],
                            [0.0, 0.0, 1.0]])
        # to center the scale
        #
        #

        t_translation = np.identity(3)
        t_translation[0, 2] = tx
        t_translation[1, 2] = ty

        t_perspective = np.eye(3)
        t_perspective[2, 0] = perspective_factor[0]
        t_perspective[2, 1] = perspective_factor[1]

        t_mat = t_perspective @ t_scale @ t_rot @ t_shear @ t_mat @ t_translation

        t_mat[0, 2] += self.pad_amount
        t_mat[1, 2] += self.pad_amount

        return t_mat
