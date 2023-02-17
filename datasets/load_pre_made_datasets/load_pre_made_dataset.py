import os.path
import glob

from datasets.listdataset import ListDataset
from utils_flow.img_processing_utils import split2list
from datasets.load_pre_made_datasets.load_data_and_add_discontinuity_dataset_with_interpolation import DiscontinuityDatasetV2


def make_dataset(dir, get_mapping, split=None, dataset_name=None):
    """
    Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm  in folder images and
      [name]_flow.flo' in folder flow """
    images = []
    if get_mapping:
        flow_dir = 'mapping'
        # flow_dir is actually mapping dir in that case, it is always normalised to [-1,1]
    else:
        flow_dir = 'flow'
    image_dir = 'images'

    # Make sure that the folders exist
    if not os.path.isdir(dir):
        raise ValueError("the training directory path that you indicated does not exist ! ")
    if not os.path.isdir(os.path.join(dir, flow_dir)):
        raise ValueError("the training directory path that you indicated does not contain the flow folder ! "
                         "Check your directories.")
    if not os.path.isdir(os.path.join(dir, image_dir)):
        raise ValueError("the training directory path that you indicated does not contain the images folder ! "
                         "Check your directories.")

    for flow_map in sorted(glob.glob(os.path.join(dir, flow_dir, '*_flow.flo'))):
        flow_map = os.path.join(flow_dir, os.path.basename(flow_map))
        root_filename = os.path.basename(flow_map)[:-9]
        img1 = os.path.join(image_dir, root_filename + '_img_1.jpg') # source image
        img2 = os.path.join(image_dir, root_filename + '_img_2.jpg') # target image
        if not (os.path.isfile(os.path.join(dir,img1)) or os.path.isfile(os.path.join(dir,img2))):
            continue
        if dataset_name is not None:
            images.append([[os.path.join(dataset_name, img1),
                            os.path.join(dataset_name, img2)],
                           os.path.join(dataset_name, flow_map)])
        else:
            images.append([[img1, img2], flow_map])
    return split2list(images, split, default_split=0.97)


def assign_default(default_dict, dict):
    if dict is None:
        dall = default_dict
    else:
        dall = {}
        dall.update(default_dict)
        dall.update(dict)
    return dall


def PreMadeDataset(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, split=None, get_mapping=False, compute_mask_zero_borders=False,
                   add_discontinuity=False, min_nbr_perturbations=5, max_nbr_perturbations=6,
                   parameters_v2=None):

    """
    Builds a dataset from existing image pairs and corresponding ground-truth flow fields and optionally add
    some flow perturbations.
    Args:
        root: path to root folder
        source_image_transform: image transformations to apply to source images
        target_image_transform: image transformations to apply to target images
        flow_transform: flow transformations to apply to ground-truth flow fields
        co_transform: transformations to apply to both images and ground-truth flow fields
        split: split (float) between training and testing, 0 means all pairs are in test_dataset
        get_mapping: output mapping instead of flow in __getittem__ ?
        compute_mask_zero_borders: output mask of zero borders ?
        add_discontinuity: add discontinuity to image pairs and corresponding ground-truth flow field ?
        min_nbr_perturbations:
        max_nbr_perturbations:
        parameters_v2: parameters of v2

    Returns:
        train_dataset
        test_dataset

    """
    perturbations_parameters_v2 = {'elastic_param': {"max_sigma": 0.04, "min_sigma": 0.1, "min_alpha": 1,
                                                     "max_alpha": 0.4},
                                   'max_sigma_mask': 10, 'min_sigma_mask': 3}

    # that is only reading and loading the data and applying transformations to both datasets
    if isinstance(root, list):
        train_list = []
        test_list = []
        for sub_root in root:
            _, dataset_name = os.path.split(sub_root)
            sub_train_list, sub_test_list = make_dataset(sub_root, get_mapping, split, dataset_name=dataset_name)
            train_list.extend(sub_train_list)
            test_list.extend(sub_test_list)
        root = os.path.dirname(sub_root)
    else:
        train_list, test_list = make_dataset(root, get_mapping, split)

    if not add_discontinuity:
        train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                    target_image_transform=target_image_transform,
                                    flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                    compute_mask_zero_borders=compute_mask_zero_borders)
        test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                                   target_image_transform=target_image_transform,
                                   flow_transform=flow_transform, co_transform=co_transform, get_mapping=get_mapping,
                                   compute_mask_zero_borders=compute_mask_zero_borders)
    else:
        parameters_v2 = assign_default(perturbations_parameters_v2, parameters_v2)
        train_dataset = DiscontinuityDatasetV2(root, train_list, source_image_transform=source_image_transform,
                                               target_image_transform=target_image_transform,
                                               flow_transform=flow_transform, co_transform=co_transform,
                                               get_mapping=get_mapping, compute_mask_zero_borders=compute_mask_zero_borders,
                                               max_nbr_perturbations=max_nbr_perturbations,
                                               min_nbr_perturbations=min_nbr_perturbations,
                                               elastic_parameters=parameters_v2['elastic_param'],
                                               max_sigma_mask=parameters_v2['max_sigma_mask'],
                                               min_sigma_mask=parameters_v2['min_sigma_mask'])
        test_dataset = DiscontinuityDatasetV2(root, test_list, source_image_transform=source_image_transform,
                                              target_image_transform=target_image_transform,
                                              flow_transform=flow_transform, co_transform=co_transform,
                                              get_mapping=get_mapping, compute_mask_zero_borders=compute_mask_zero_borders,
                                              max_nbr_perturbations=max_nbr_perturbations,
                                              min_nbr_perturbations=min_nbr_perturbations,
                                              elastic_parameters=parameters_v2['elastic_param'],
                                              max_sigma_mask=parameters_v2['max_sigma_mask'],
                                              min_sigma_mask=parameters_v2['min_sigma_mask'])
    return train_dataset, test_dataset
