r"""Implementation of Dynamic Layer Gating (DLG)"""
import torch.nn as nn
import torch


class GumbelFeatureSelection(nn.Module):
    r"""Dynamic layer gating with Gumbel-max trick"""
    def __init__(self, in_channels, reduction=8, hidden_size=32):
        r"""Constructor for DLG"""
        super(GumbelFeatureSelection, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.reduction = reduction

        # Learnable modules in Dynamic Hyperpixel Flow
        self.reduction_ffns = []  # Convolutional Feature Transformation (CFT)
        self.gumbel_ffns = []  # Gumbel Layer Gating (GLG)
        for in_channel in in_channels:
            out_channel = in_channel // self.reduction
            reduction_ffn = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
            gumbel_ffn = nn.Sequential(
                nn.Conv2d(in_channel, hidden_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False)
            )
            self.reduction_ffns.append(reduction_ffn)
            self.gumbel_ffns.append(gumbel_ffn)
        self.reduction_ffns = nn.ModuleList(self.reduction_ffns)
        self.gumbel_ffns = nn.ModuleList(self.gumbel_ffns)

    def forward(self, lid, src_feat, trg_feat):
        r"""DLG forward pass"""
        relevance = self.gumbel_ffns[lid](self.avgpool(src_feat) + self.avgpool(trg_feat))
        # For measuring per-pair inference time on test set
        if not self.training and len(relevance) == 1:  # batch size equal to 1
            selected = relevance.max(dim=1)[1].squeeze()

            # Perform CFT iff the layer is selected
            if selected:
                src_x = self.reduction_ffns[lid](src_feat)
                trg_x = self.reduction_ffns[lid](trg_feat)
            else:
                src_x = None
                trg_x = None
            layer_sel = relevance.view(-1).max(dim=0)[1].unsqueeze(0).float()
        else:
            # Hard selection during forward pass (layer_sel)
            # Soft gradients during backward pass (dL/dy)
            y = self.gumbel_softmax(relevance.squeeze())
            _y = self._softmax(y)
            layer_sel = y[:, 1] + _y[:, 1]

            src_x = self.reduction_ffns[lid](src_feat)
            trg_x = self.reduction_ffns[lid](trg_feat)
            src_x = src_x * y[:, 1].view(-1, 1, 1, 1) + src_x * _y[:, 1].view(-1, 1, 1, 1)
            trg_x = trg_x * y[:, 1].view(-1, 1, 1, 1) + trg_x * _y[:, 1].view(-1, 1, 1, 1)

        return src_x, trg_x, layer_sel

    def _softmax(self, soft_sample):
        r"""Gumbel-max trick: replaces argmax with softmax during backward pass: soft_sample + _soft_sample"""
        hard_sample_idx = torch.max(soft_sample, dim=1)[1].unsqueeze(1)
        hard_sample = soft_sample.detach().clone().zero_().scatter(dim=1, index=hard_sample_idx.long(),
                                                                   value=1.0)

        _soft_sample = (hard_sample - soft_sample.detach().clone())

        return _soft_sample

    def gumbel_softmax(self, logits, temperature=1, eps=1e-10):
        """Softly draws a sample from the Gumbel distribution"""
        if self.training:
            gumbel_noise = -torch.log(eps - torch.log(logits.detach().clone().uniform_() + eps))
            gumbel_input = logits + gumbel_noise
        else:
            gumbel_input = logits

        if gumbel_input.dim() == 1:
            gumbel_input = gumbel_input.unsqueeze(0)

        soft_sample = self.softmax(gumbel_input / temperature)

        return soft_sample
