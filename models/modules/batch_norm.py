
import torch
import torch.nn as nn


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)

    def forward(self, input):
        if self.training:
            return super().forward(input)
        else:

            if self.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.momentum
            scaling = self.weight / (torch.sqrt(self.running_var + self.eps))
            output = input * scaling.view(-1,1,1) + (self.bias - self.running_mean * scaling).view(-1,1,1)
            return output