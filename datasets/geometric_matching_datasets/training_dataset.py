"""
Extracted from DGC-Net https://github.com/AaltoVision/DGC-Net/blob/master/data/dataset.py and modified
"""
from os import path as osp
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.utils.data import Dataset


from utils_flow.img_processing_utils import center_crop
from utils_flow.flow_and_mapping_operations import unormalise_and_convert_mapping_to_flow
from utils_flow.img_processing_utils import define_mask_zero_borders


class HomoAffTpsDataset(Dataset):
    """
    Extracted from https://github.com/AaltoVision/DGC-Net/blob/master/data/dataset.py and modified
    Main dataset for generating the training/validation the proposed approach.
    It can handle affine, TPS, and Homography transformations.
    """

    def __init__(self, image_path, csv_file, transforms, transforms_target=None, get_flow=False,
                 compute_mask_zero_borders=False, pyramid_param=[520], output_image_size=(520, 520),
                 original_DGC_transfo=True):
        """
        Args:
            image_path: filepath to the dataset
            csv_file: csv file with ground-truth transformation parameters and name of original images
            transforms: image transformations for the source image (data preprocessing)
            transforms_target: image transformations for the target image (data preprocessing), if different than that of
            the source image
            get_flow: bool, whether to get flow or normalized mapping
            compute_mask_zero_borders: bool mask removing the black image borders
            pyramid_param: spatial resolution of the feature maps at each level
                of the feature pyramid (list)
            output_image_size: size (tuple) of the output images
        Output:
            if get_flow:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                flow_map: corresponding ground-truth flow field, shape 2xHxW
                correspondence_mask: mask of valid flow, shape HxW
            else:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                correspondence_map: correspondence_map, normalized to [-1,1], shape HxWx2,
                                    should correspond to correspondence_map_pyro[-1]
                correspondence_map_pyro: pixel correspondence map for each feature pyramid level
                mask_x: X component of the mask (valid/invalid correspondences)
                mask_y: Y component of the mask (valid/invalid correspondences)
                correspondence_mask: mask of valid flow, shape HxW, equal to mask_x and mask_y
        """
        super().__init__()
        self.img_path = image_path
        self.mask_zero_borders = compute_mask_zero_borders
        if not os.path.isdir(self.img_path):
            raise ValueError("The image path that you indicated does not exist!")

        self.transform_dict = {0: 'aff', 1: 'tps', 2: 'homo'}
        self.transforms_source = transforms
        if transforms_target is None:
            self.transforms_target = transforms
        else:
            self.transforms_target = transforms_target
        self.pyramid_param = pyramid_param
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            if len(self.df) == 0:
                raise ValueError("The csv file that you indicated is empty !")
        else:
            raise ValueError("The path to the csv file that you indicated does not exist !")
        self.get_flow = get_flow
        self.H_OUT, self.W_OUT = output_image_size

        # changed compared to version from DGC-Net
        self.ratio_cropping = 1.5
        # this is a scaling to apply to the homographies, usually applied to get 240x240 images
        if original_DGC_transfo:
            self.ratio_TPS = self.H_OUT / 240.0
            self.ratio_homography = self.H_OUT / 240.0

            self.H_AFF_TPS, self.W_AFF_TPS = (int(480*self.ratio_TPS), int(640*self.ratio_TPS))
            self.H_HOMO, self.W_HOMO = (int(576*self.ratio_homography), int(768*self.ratio_homography))
        else:
            self.ratio_TPS = 950/520
            self.ratio_homography = 950.0/520.0
            self.H_AFF_TPS, self.W_AFF_TPS = (int(480*self.ratio_TPS), int(640*self.ratio_TPS))
            self.H_HOMO, self.W_HOMO = (int(950), int(950))

        self.THETA_IDENTITY = \
            torch.Tensor(np.expand_dims(np.array([[1, 0, 0],
                                                  [0, 1, 0]]),
                                        0).astype(np.float32))
        self.gridGen = TpsGridGen(self.H_OUT, self.W_OUT)

    def transform_image(self,
                        image,
                        out_h,
                        out_w,
                        padding_factor=1.0,
                        crop_factor=1.0,
                        theta=None):
        sampling_grid = self.generate_grid(out_h, out_w, theta)
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image

        if version.parse(torch.__version__) >= version.parse("1.3"):
            warped_image_batch = F.grid_sample(image, sampling_grid, align_corners=True)
        else:
            warped_image_batch = F.grid_sample(image, sampling_grid)

        return warped_image_batch

    def generate_grid(self, out_h, out_w, theta=None):
        out_size = torch.Size((1, 3, out_h, out_w))
        if theta is None:
            theta = self.THETA_IDENTITY
            theta = theta.expand(1, 2, 3).contiguous()
            return F.affine_grid(theta, out_size)
        elif (theta.shape[1] == 2):
            return F.affine_grid(theta, out_size)
        else:
            return self.gridGen(theta)

    def get_grid(self, H, ccrop):
        # top-left corner of the central crop
        X_CCROP, Y_CCROP = ccrop[0], ccrop[1]

        W_FULL, H_FULL = (self.W_HOMO, self.H_HOMO)
        W_SCALE, H_SCALE = (self.W_OUT, self.H_OUT)

        # inverse homography matrix
        Hinv = np.linalg.inv(H)
        Hscale = np.eye(3)
        Hscale[0,0] = Hscale[1,1] = self.ratio_homography
        Hinv = Hscale @ Hinv @ np.linalg.inv(Hscale)

        # estimate the grid for the whole image
        X, Y = np.meshgrid(np.linspace(0, W_FULL - 1, W_FULL),
                           np.linspace(0, H_FULL - 1, H_FULL))
        X_, Y_ = X, Y
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.dot(Hinv, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        X_grid_pivot = (XwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)
        Y_grid_pivot = (YwarpHom / (ZwarpHom + 1e-8)).view(H_FULL, W_FULL)

        # normalize XwarpHom and YwarpHom and cast to [-1, 1] range
        Xwarp = (2 * X_grid_pivot / (W_FULL - 1) - 1)
        Ywarp = (2 * Y_grid_pivot / (H_FULL - 1) - 1)
        grid_full = torch.stack([Xwarp, Ywarp], dim=-1)

        # getting the central patch from the pivot
        Xwarp_crop = X_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        Ywarp_crop = Y_grid_pivot[Y_CCROP:Y_CCROP + H_SCALE,
                                  X_CCROP:X_CCROP + W_SCALE]
        X_crop = X_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]
        Y_crop = Y_[Y_CCROP:Y_CCROP + H_SCALE,
                    X_CCROP:X_CCROP + W_SCALE]

        # crop grid
        Xwarp_crop_range = \
            2 * (Xwarp_crop - X_crop.min()) / (X_crop.max() - X_crop.min()) - 1
        Ywarp_crop_range = \
            2 * (Ywarp_crop - Y_crop.min()) / (Y_crop.max() - Y_crop.min()) - 1
        grid_crop = torch.stack([Xwarp_crop_range,
                                 Ywarp_crop_range], dim=-1)
        return grid_full.unsqueeze(0), grid_crop.unsqueeze(0)

    @staticmethod
    def symmetric_image_pad(image_batch, padding_factor):
        """
        Pad an input image mini-batch symmetrically
        Args:
            image_batch: an input image mini-batch to be pre-processed
            padding_factor: padding factor
        Output:
            image_batch: padded image mini-batch
        """
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))

        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left),
                                 image_batch,
                                 image_batch.index_select(3, idx_pad_right)),
                                3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top),
                                 image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)),
                                2)
        return image_batch

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:
            if get_flow:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                flow_map: corresponding ground-truth flow field, shape 2xHxW
                correspondence_mask: mask of valid flow, shape HxW
            else:
                source_image: source image, shape 3xHxWx
                target_image: target image, shape 3xHxWx
                correspondence_map: correspondence_map, normalized to [-1,1], shape HxWx2,
                                    should correspond to correspondence_map_pyro[-1]
                correspondence_map_pyro: pixel correspondence map for each feature pyramid level
                mask_x: X component of the mask (valid/invalid correspondences)
                mask_y: Y component of the mask (valid/invalid correspondences)
                correspondence_mask: mask of valid flow, shape HxW, equal to mask_x and mask_y

        """
        data = self.df.iloc[idx]
        # get the transformation type flag
        transform_type = data['aff/tps/homo'].astype('uint8')
        source_img_name = osp.join(self.img_path, data.fname)
        if not os.path.exists(source_img_name):
            raise ValueError("The path to one of the original image {} does not exist, check your image path "
                             "and your csv file !".format(source_img_name))

        # aff/tps transformations
        if transform_type == 0 or transform_type == 1:
            # read image
            source_img = cv2.cvtColor(cv2.imread(source_img_name),
                                      cv2.COLOR_BGR2RGB)

            # cropping dimention of the image first if it is too big, would occur to big resizing after
            if source_img.shape[0] > self.H_AFF_TPS*self.ratio_cropping or \
               source_img.shape[1] > self.W_AFF_TPS*self.ratio_cropping:
                source_img, x, y = center_crop(source_img, (int(self.W_AFF_TPS*self.ratio_cropping),
                                                            int(self.H_AFF_TPS*self.ratio_cropping)))

            if transform_type == 0:
                theta = data.iloc[2:8].values.astype('float').reshape(2, 3)
                theta = torch.Tensor(theta.astype(np.float32)).expand(1, 2, 3)
            else:
                theta = data.iloc[2:].values.astype('float')
                theta = np.expand_dims(np.expand_dims(theta, 1), 2)
                theta = torch.Tensor(theta.astype(np.float32))
                theta = theta.expand(1, 18, 1, 1)

            # make arrays float tensor for subsequent processing
            image = torch.Tensor(source_img.astype(np.float32))

            if image.numpy().ndim == 2:
                image = \
                    torch.Tensor(np.dstack((source_img.astype(np.float32),
                                            source_img.astype(np.float32),
                                            source_img.astype(np.float32))))
            image = image.transpose(1, 2).transpose(0, 1)

            # Resize image using bilinear sampling with identity affine
            image_pad = self.transform_image(image.unsqueeze(0), self.H_AFF_TPS, self.W_AFF_TPS)

            # padding and crop factor depend where to crop and pad image
            img_src_crop = \
                self.transform_image(image_pad,
                                     self.H_OUT,
                                     self.W_OUT,
                                     padding_factor=0.8,
                                     crop_factor=9/16).squeeze().numpy()

            img_target_crop = \
                self.transform_image(image_pad,
                                     self.H_OUT,
                                     self.W_OUT,
                                     padding_factor=0.8,
                                     crop_factor=9/16,
                                     theta=theta).squeeze().numpy()

            # convert to [H, W, C] convention (for np arrays)
            img_src_crop = img_src_crop.transpose((1, 2, 0))
            img_target_crop = img_target_crop.transpose((1, 2, 0))

        # Homography transformation
        elif transform_type == 2:

            # ATTENTION CV2 resize is inverted, first w and then h
            theta = data.iloc[2:11].values.astype('double').reshape(3, 3)
            source_img = cv2.cvtColor(cv2.imread(source_img_name), cv2.COLOR_BGR2RGB)

            # cropping dimention of the image first if it is too big, would occur to big resizing after
            if source_img.shape[0] > self.H_HOMO * self.ratio_cropping \
                    or source_img.shape[1] > self.W_HOMO*self.ratio_cropping:
                source_img, x, y = center_crop(source_img, (int(self.W_HOMO*self.ratio_cropping),
                                                            int(self.H_HOMO*self.ratio_cropping)))

            # resize to value stated at the beginning
            img_src_orig = cv2.resize(source_img, dsize=(self.W_HOMO, self.H_HOMO),
                                      interpolation=cv2.INTER_LINEAR) # cv2.resize, W is giving first

            # get a central crop:
            img_src_crop, x1_crop, y1_crop = center_crop(img_src_orig,
                                                         self.W_OUT)

            # Obtaining the full and crop grids out of H
            grid_full, grid_crop = self.get_grid(theta,
                                                 ccrop=(x1_crop, y1_crop))

            # warp the fullsize original source image
            img_src_orig = torch.Tensor(img_src_orig.astype(np.float32))
            img_src_orig = img_src_orig.permute(2, 0, 1)

            if version.parse(torch.__version__) >= version.parse("1.3"):
                img_orig_target_vrbl = F.grid_sample(img_src_orig.unsqueeze(0),
                                                     grid_full, align_corners=True)
            else:
                img_orig_target_vrbl = F.grid_sample(img_src_orig.unsqueeze(0),
                                                     grid_full)

            img_orig_target_vrbl = \
                img_orig_target_vrbl.squeeze().permute(1, 2, 0)

            # get the central crop of the target image
            img_target_crop, _, _ = center_crop(img_orig_target_vrbl.numpy(),
                                                self.W_OUT)

        else:
            print('Error: transformation type')

        if self.mask_zero_borders:
            mask_valid = define_mask_zero_borders(img_target_crop)

        if self.transforms_source is not None and self.transforms_target is not None:
            cropped_source_image = \
                self.transforms_source(img_src_crop.astype(np.uint8))
            cropped_target_image = \
                self.transforms_target(img_target_crop.astype(np.uint8))
        else:
            # if no specific transformations are applied, they are just put in 3xHxW
            cropped_source_image = \
                torch.Tensor(img_src_crop.astype(np.float32))
            cropped_target_image = \
                torch.Tensor(img_target_crop.astype(np.float32))

            # convert to [C, H, W] convention (for tensors)
            cropped_source_image = cropped_source_image.permute(-1, 0, 1)
            cropped_target_image = cropped_target_image.permute(-1, 0, 1)

        # construct a pyramid with a corresponding grid on each layer
        grid_pyramid = []
        mask_x = []
        mask_y = []
        if transform_type == 0:
            for layer_size in self.pyramid_param:
                # get layer size or change it so that it corresponds to PWCNet
                grid = self.generate_grid(layer_size,
                                          layer_size,
                                          theta).squeeze(0)
                mask = grid.ge(-1) & grid.le(1)
                grid_pyramid.append(grid)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])
        elif transform_type == 1:
            grid = self.generate_grid(self.H_OUT,
                                      self.W_OUT,
                                      theta).squeeze(0)
            for layer_size in self.pyramid_param:
                grid_m = torch.from_numpy(cv2.resize(grid.numpy(),
                                                     (layer_size, layer_size)))
                mask = grid_m.ge(-1) & grid_m.le(1)
                grid_pyramid.append(grid_m)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])
        elif transform_type == 2:
            grid = grid_crop.squeeze(0)
            for layer_size in self.pyramid_param:
                grid_m = torch.from_numpy(cv2.resize(grid.numpy(),
                                                    (layer_size, layer_size)))
                mask = grid_m.ge(-1) & grid_m.le(1)
                grid_pyramid.append(grid_m)
                mask_x.append(mask[:, :, 0])
                mask_y.append(mask[:, :, 1])

        output = {'source_image': cropped_source_image,  'target_image': cropped_target_image,
                  'correspondence_mask': np.logical_and(mask_x[-1].detach().numpy(), mask_y[-1].detach().numpy())
                      .astype(np.bool if version.parse(torch.__version__) >= version.parse("1.1")
                              else np.uint8), 'sparse': False}

        if self.mask_zero_borders:
            output['mask_zero_borders']= mask_valid.astype(np.bool) if \
                version.parse(torch.__version__) >= version.parse("1.1") else mask_valid.astype(np.uint8)

        if self.get_flow:
            # ATTENTION, here we just get the flow of the highest resolution asked, not the pyramid of flows !
            flow = unormalise_and_convert_mapping_to_flow(grid_pyramid[-1], output_channel_first=True)
            output['flow_map'] = flow  # here flow map is 2 x h x w]
        else:
            output['correspondence_map'] = grid_pyramid[-1],  # torch tensor,  h x w x 2
            output['correspondence_map_pyro'] = grid_pyramid
        return output


class TpsGridGen(nn.Module):
    """
    Adopted version of synthetically transformed pairs dataset by I.Rocco
    https://github.com/ignacio-rocco/cnngeometric_pytorch
    """

    def __init__(self,
                 out_h=240,
                 out_w=240,
                 use_regular_grid=True,
                 grid_size=3,
                 reg_factor=0,
                 use_cuda=False):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
                                               np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = \
                P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            self.P_Y = \
                P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta,
                                                torch.cat((self.grid_X,
                                                           self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        # num of points (along dim 0)
        N = X.size()[0]

        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = \
            torch.pow(Xmat - Xmat.transpose(0, 1), 2) + \
            torch.pow(Ymat - Ymat.transpose(0, 1), 2)

        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))

        # construct matrix L
        OO = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((OO, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1),
                       torch.cat((P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        '''
        points should be in the [B,H,W,2] format,
        where points[:,:,:,0] are the X coords
        and points[:,:,:,1] are the Y coords
        '''

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        '''
        repeat pre-defined control points along
        spatial dimensions of points to be transformed
        '''
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_X)
        W_Y = \
            torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
                                                           self.N,
                                                           self.N)), Q_Y)
        '''
        reshape
        W_X,W,Y: size [B,H,W,1,N]
        '''
        W_X = \
            W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        W_Y = \
            W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                 points_h,
                                                                 points_w,
                                                                 1,
                                                                 1)
        # compute weights for affine part
        A_X = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_X)
        A_Y = \
            torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size,
                                                           3,
                                                           self.N)), Q_Y)
        '''
        reshape
        A_X,A,Y: size [B,H,W,1,3]
        '''
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
                                                                   points_h,
                                                                   points_w,
                                                                   1,
                                                                   1)
        '''
        compute distance P_i - (grid_X,grid_Y)
        grid is expanded in point dim 4, but not in batch dim 0,
        as points P_X,P_Y are fixed for all batch
        '''
        sz_x = points[:, :, :, 0].size()
        sz_y = points[:, :, :, 1].size()
        p_X_for_summation = points[:, :, :, 0].unsqueeze(3).unsqueeze(4)
        p_X_for_summation = p_X_for_summation.expand(sz_x + (1, self.N))
        p_Y_for_summation = points[:, :, :, 1].unsqueeze(3).unsqueeze(4)
        p_Y_for_summation = p_Y_for_summation.expand(sz_y + (1, self.N))

        if points_b == 1:
            delta_X = p_X_for_summation - P_X
            delta_Y = p_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = p_X_for_summation - P_X.expand_as(p_X_for_summation)
            delta_Y = p_Y_for_summation - P_Y.expand_as(p_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        '''
        U: size [1,H,W,1,N]
        avoid NaN in log computation
        '''
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                                                   points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                                                   points_Y_batch.size()[1:])

        points_X_prime = \
            A_X[:, :, :, :, 0] + \
            torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = \
            A_Y[:, :, :, :, 0] + \
            torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
            torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
            torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)
