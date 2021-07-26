import os
import numpy as np
import argparse
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
env_path = os.path.join(os.path.dirname(__file__), '../')
if env_path not in sys.path:
    sys.path.append(env_path)
from utils_data.io import boolean_string
from datasets.geometric_matching_datasets.training_dataset import HomoAffTpsDataset
from utils_flow.pixel_wise_mapping import remap_using_flow_fields

from utils_data.image_transforms import ArrayToTensor
from utils_data.io import writeFlow

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DGC-Net train script')
    parser.add_argument('--image_data_path', type=str,
                        help='path to directory containing the original images.')
    parser.add_argument('--csv_path', type=str, default='datasets/csv_files/homo_aff_tps_train_DPED_CityScape_ADE.csv',
                        help='path to the CSV files')
    parser.add_argument('--save_dir', type=str,
                        help='path directory to save the image pairs and corresponding ground-truth flows')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot as examples the first 4 pairs ? default is False')
    parser.add_argument('--seed', type=int, default=1981,
                        help='Pseudo-RNG seed')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    plot = args.plot
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image_dir=os.path.join(save_dir, 'images')
    flow_dir = os.path.join(save_dir, 'flow')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    # datasets
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    pyramid_param = [520]

    # training dataset
    train_dataset = HomoAffTpsDataset(image_path=args.image_data_path,
                                      csv_file=args.csv_path,
                                      transforms=source_img_transforms,
                                      transforms_target=target_img_transforms,
                                      pyramid_param=pyramid_param,
                                      get_flow=True,
                                      output_image_size=(520, 520))

    test_dataloader = DataLoader(train_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, minibatch in pbar:
        image_source = minibatch['source_image'] # shape is 1x3xHxW
        image_target = minibatch['target_image']
        if image_source.shape[1] == 3:
            image_source = image_source.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            image_source = image_source[0].numpy().astype(np.uint8)

        if image_target.shape[1] == 3:
            image_target = image_target.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            image_target = image_target[0].numpy().astype(np.uint8)

        flow_gt = minibatch['flow_map'][0].permute(1,2,0).numpy() # now shape is HxWx2

        # save the flow file and the images files
        base_name = 'image_{}'.format(i)
        name_flow = base_name + '_flow.flo'
        writeFlow(flow_gt, name_flow, flow_dir)
        imageio.imwrite(os.path.join(save_dir, 'images/', base_name + '_img_1.jpg'), image_source)
        imageio.imwrite(os.path.join(save_dir, 'images/', base_name + '_img_2.jpg'), image_target)

        # plotting to make sure that eevrything is working
        if plot and i < 4:
            # just for now
            fig, axis = plt.subplots(1, 3, figsize=(20, 20))
            axis[0].imshow(image_source)
            axis[0].set_title("Image source")
            axis[1].imshow(image_target)
            axis[1].set_title("Image target")
            remapped_gt = remap_using_flow_fields(image_source, flow_gt[:,:,0], flow_gt[:,:,1])

            axis[2].imshow(remapped_gt)
            axis[2].set_title("Warped source image according to ground truth flow")
            fig.savefig(os.path.join(save_dir, 'synthetic_pair_{}'.format(i)), bbox_inches='tight')
            plt.close(fig)