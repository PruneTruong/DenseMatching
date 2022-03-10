import torch
import torch.nn as nn


class LearntBinParam(nn.Module):
    """Class to introduce a learnt bin/occluded state in the cost volume. """
    def __init__(self, initial_value=1.0):
        super().__init__()
        self.bin_score = torch.nn.Parameter(torch.tensor(initial_value))

    def forward(self, correlation, *args, **kwargs):
        if len(correlation.shape) == 3:
            b, c, hw = correlation.shape
            # if it is from target to source, then shape is b, h_s*w_s, h_t*w_t

            bins0 = self.bin_score.expand(b, 1, hw)
        else:
            b = correlation.shape[0]
            h, w = correlation.shape[-2:]
            correlation = correlation.view(b, -1, h, w)
            # if it is from target to source, then shape is b, h_s*w_s, h_t, w_t

            bins0 = self.bin_score.expand(b, 1, h, w)
        aug_corr = torch.cat((correlation, bins0.to(correlation.device)), 1)   # b, c+1, h, w
        return aug_corr
