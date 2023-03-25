import os
import cv2
import random
import torch
from pycocotools.coco import COCO
from collections import OrderedDict
from torch.utils.data import Dataset

from datasets.object_augmented_dataset.base_image_dataset import BaseImageDataset
from datasets.object_augmented_dataset.image_loader import jpeg4py_loader, opencv_loader
from utils_flow.img_processing_utils import pad_to_size, resize_keeping_aspect_ratio


class MSCOCOImages(Dataset):
    """Dataset loading images from COCO Dataset.
    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
                - instances_train2017.json
            - images
                - train2014
                - train2017
    """
    def __init__(self, root, image_transform=None, output_image_size=(520, 520), split='train'):
        """

        Args:
            root: root directory containing the images folder
            image_transform: image transformations to apply
            output_image_size: output image size. If (h, w), resize to hxw. If only a single scalar is provided, resize
                               so that the largest dimension is equal to output_image_size, and pad the other
                               dimension with zero to output_image_size. Final output size is then
                               (output_image_size, output_image_size).
            split: 'train' or 'val'.

        Output in __getitem__:
            image
        """

        self.root = root
        self.image_transform = image_transform
        self.output_image_size = output_image_size
        self.split = split

        self.sample_items()
        print('COCO: {} dataset comprises {} image pairs'.format(self.split, self.__len__()))

    def sample_items(self):
        self.items = []
        root_image = os.path.join(self.root, 'images')
        folders = [f for f in sorted(os.listdir(root_image)) if os.path.isdir(os.path.join(root_image, f)) and self.split in f]
        for folder in folders:
            for image in [f for f in sorted(os.listdir(os.path.join(root_image, folder))) if f.endswith('.jpg')]:
                self.items.append(os.path.join('images', folder, image))

    def __len__(self):
        return len(self.items)

    def _read_single_view(self, idx):
        path = os.path.join(self.root, self.items[idx])
        image = cv2.imread(path)
        if len(image.shape) != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = image[:, :, ::-1]  # go from BGR to RGB

        if self.output_image_size is not None:
            if isinstance(self.output_image_size, list):
                # resize to a fixed size and rescale the keypoints accordingly
                image = cv2.resize(image, (self.output_image_size[1], self.output_image_size[0]))

            else:
                # rescale both images so that the largest dimension is equal to the desired size of image and
                # then pad to obtain size 256x256 or whatever desired size.
                image, ratio_ = resize_keeping_aspect_ratio(image, self.output_image_size)
                image = pad_to_size(image, self.output_image_size)

        if self.image_transform is not None:
            image = self.image_transform(image)

        return {'image': image}

    def __getitem__(self, idx):
        """
        Args:
            idx

        Returns: Dictionary with fieldname:
            image
        """
        output = self._read_single_view(idx)
        return output


class MSCOCO(BaseImageDataset):
    """ The COCO object detection dataset.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
                - instances_train2017.json
            - images
                - train2014
                - train2017

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None, min_area=None,
                 split="train", version="2014"):
        """
        args:
            root - path to coco root folder
            image_loader (jpeg4py_loader) - The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
            min_area - Objects with area less than min_area are filtered out. Default is 0.0
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
        """

        super().__init__('COCO', root, image_loader)

        self.img_pth = os.path.join(root, 'images/{}{}/'.format(split, version))
        self.anno_path = os.path.join(root, 'annotations/instances_{}{}.json'.format(split, version))

        self.coco_set = COCO(self.anno_path)

        self.cats = self.coco_set.cats

        self.class_list = self.get_class_list()  # the parent class thing would happen in the sampler

        self.image_list = self._get_image_list(min_area=min_area)

        if data_fraction is not None:
            self.image_list = random.sample(self.image_list, int(len(self.image_list) * data_fraction))
        self.im_per_class = self._build_im_per_class()

    def _get_image_list(self, min_area=None):
        ann_list = list(self.coco_set.anns.keys())
        image_list = [a for a in ann_list if self.coco_set.anns[a]['iscrowd'] == 0]

        if min_area is not None:
            image_list = [a for a in image_list if self.coco_set.anns[a]['area'] > min_area]

        return image_list

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'coco'

    def has_class_info(self):
        return True

    def has_segmentation_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def _build_im_per_class(self):
        im_per_class = {}
        for i, im in enumerate(self.image_list):
            class_name = self.cats[self.coco_set.anns[im]['category_id']]['name']
            if class_name not in im_per_class:
                im_per_class[class_name] = [i]
            else:
                im_per_class[class_name].append(i)

        return im_per_class

    def get_images_in_class(self, class_name):
        return self.im_per_class[class_name]

    def get_image_info(self, im_id):
        anno = self._get_anno(im_id)

        bbox = torch.Tensor(anno['bbox']).view(4,)

        mask = torch.Tensor(self.coco_set.annToMask(anno))

        valid = (bbox[2] > 0) & (bbox[3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}

    def _get_anno(self, im_id):
        anno = self.coco_set.anns[self.image_list[im_id]]

        return anno

    def _get_image(self, im_id):
        path = self.coco_set.loadImgs([self.coco_set.anns[self.image_list[im_id]]['image_id']])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        if len(img.shape) == 2:
            # black and white image
            img = cv2.cvtColor(img, cv2. COLOR_GRAY2RGB)
        return img

    def get_meta_info(self, im_id):
        try:
            cat_dict_current = self.cats[self.coco_set.anns[self.image_list[im_id]]['category_id']]
            object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def get_class_name(self, im_id):
        cat_dict_current = self.cats[self.coco_set.anns[self.image_list[im_id]]['category_id']]
        return cat_dict_current['name']

    def get_image(self, image_id, anno=None):
        frame = self._get_image(image_id)

        if anno is None:
            anno = self.get_image_info(image_id)

        object_meta = self.get_meta_info(image_id)

        return frame, anno, object_meta
