import torch
import torch.nn as nn


class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    @staticmethod
    def forward(feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(feature**2, 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return feature / norm


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
    return feature / norm


def compute_global_correlation(feature_source, feature_target, shape='3D', put_W_first_in_channel_dimension=False):
    if shape == '3D':
        b, c, h_source, w_source = feature_source.size()
        b, c, h_target, w_target = feature_target.size()

        if put_W_first_in_channel_dimension:
            # FOR SOME REASON, THIS IS THE DEFAULT
            feature_source = feature_source.transpose(2, 3).contiguous().view(b, c, w_source * h_source)
            # ATTENTION, here the w and h of the source features are inverted !!!
            # shape (b,c, w_source * h_source)

            feature_target = feature_target.view(b, c, h_target * w_target).transpose(1, 2)
            # shape (b,h_target*w_target,c)

            # perform matrix mult.
            feature_mul = torch.bmm(feature_target, feature_source)
            # shape (b,h_target*w_target, w_source*h_source)
            # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]

            correlation_tensor = feature_mul.view(b, h_target, w_target, w_source * h_source).transpose(2, 3) \
                .transpose(1, 2)
            # shape (b, w_source*h_source, h_target, w_target)
            # ATTENTION, here in source dimension, W is first !! (channel dimension is W,H)
        else:
            feature_source = feature_source.contiguous().view(b, c, h_source * w_source)
            # shape (b,c, h_source * w_source)

            feature_target = feature_target.view(b, c, h_target * w_target).transpose(1, 2)
            # shape (b,h_target*w_target,c)

            # perform matrix mult.
            feature_mul = torch.bmm(feature_target,
                                    feature_source)  # shape (b,h_target*w_target, h_source*w_source)
            correlation_tensor = feature_mul.view(b, h_target, w_target, h_source * w_source).transpose(2, 3) \
                .transpose(1, 2)
            # shape (b, h_source*w_source, h_target, w_target)
            # ATTENTION, here H is first in channel dimension !
    elif shape == '4D':
        b, c, hsource, wsource = feature_source.size()
        b, c, htarget, wtarget = feature_target.size()
        # reshape features for matrix multiplication
        feature_source = feature_source.view(b, c, hsource * wsource).transpose(1, 2)  # size [b,hsource*wsource,c]
        feature_target = feature_target.view(b, c, htarget * wtarget)  # size [b,c,htarget*wtarget]
        # perform matrix mult.
        feature_mul = torch.bmm(feature_source, feature_target)  # size [b, hsource*wsource, htarget*wtarget]
        correlation_tensor = feature_mul.view(b, hsource, wsource, htarget, wtarget).unsqueeze(1)
        # size is [b, 1, hsource, wsource, htarget, wtarget]
    else:
        raise ValueError('tensor should be 3D or 4D')

    return correlation_tensor


class GlobalFeatureCorrelationLayer(torch.nn.Module):
    """
    Implementation of the global feature correlation layer
    Source and query, as well as target and reference refer to the same images.
    """
    def __init__(self, shape='3D', normalization=False, put_W_first_in_channel_dimension=False):
        super(GlobalFeatureCorrelationLayer, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()
        self.put_W_first_in_channel_dimension = put_W_first_in_channel_dimension

    def forward(self, feature_source, feature_target):
        correlation_tensor = compute_global_correlation(feature_source, feature_target, shape=self.shape,
                                                        put_W_first_in_channel_dimension=
                                                        self.put_W_first_in_channel_dimension)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor
