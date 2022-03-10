import torch


def build_one_hot(index, dim):
    res = torch.zeros(dim)
    res[index] = 1
    return res


class OneHotCrossEntropy(torch.nn.Module):
    """ Cross entropy loss, with a single label. """
    def __init__(self, reduction="mean", log2=False):
        super(OneHotCrossEntropy, self).__init__()
        self.reduction = reduction
        self.log2 = log2

    def forward(self, logits, index_of_target, mask=None, *args, **kwargs):
        """
        Args:
            logits: already softmaxed, NxC
            index_of_target: index of the target [0, C], shape is N
            mask:
            *args:
            **kwargs:
        """
        assert len(index_of_target.shape) == 1
        assert len(logits.shape) == 2

        if self.log2:
            logits = torch.log2(logits + 1e-7)
        else:
            logits = torch.log(logits + 1e-7)

        index = torch.arange(0, logits.shape[0])
        loss = - logits[index, index_of_target]
        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()


class SmoothCrossEntropy(torch.nn.Module):
    """ Cross entropy loss, with a smooth probability map as ground-truth """
    def __init__(self, reduction='mean_per_el'):
        super(SmoothCrossEntropy, self).__init__()

    @staticmethod
    def _ce_impl(input, target):
        input = torch.log(input + 1e-7)
        output_pos = - target * input
        zeros_ = torch.zeros_like(output_pos)
        mask_ = output_pos.gt(0)
        # we only keep probabilities that are higher than 0,
        # will also remove nan case when gt and prediction is 0=> log(0) -infini
        output = torch.where(mask_, output_pos, zeros_)
        return output

    def forward(self, logits, target, *args, **kwargs):
        """
        Args:
            logits: already softmaxed, NxC
            target: gt probability map, shape is NxC
            *args:
            **kwargs:
        """

        output = self._ce_impl(torch.clip(logits, 0, 1), target).sum()
        output = output / logits.shape[0]
        return output


class BinaryCrossEntropy(torch.nn.BCELoss):
    """ Binary cross entropy loss, with a smooth probability map as ground-truth.
    Binary cross entropy is enforced for each element. """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean_per_el'):
        reduction_ = reduction
        if reduction == 'mean_per_el':
            reduction_ = 'sum'
        self.reduction_name = reduction
        super(BinaryCrossEntropy, self).__init__(weight, size_average, reduce, reduction_)

    def forward(self, logits, target, *args, **kwargs):
        """
        Args:
            logits: already softmaxed, NxC
            target: gt probability map, shape is NxC
            *args:
            **kwargs:
        """
        output = super().forward(torch.clip(logits, 0, 1), target)
        if self.reduction_name == 'mean_per_el':
            output = output / logits.shape[0]
        return output


class OneHotBinaryCrossEntropy(torch.nn.BCELoss):
    """ Binary cross entropy loss, with a one-hot probability map as ground-truth, i.e. there is only one label.
    Binary cross entropy is enforced for each element. """
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean_per_el'):
        reduction_ = reduction
        if reduction == 'mean_per_el':
            reduction_ = 'sum'
        self.reduction_name = reduction
        super(OneHotBinaryCrossEntropy, self).__init__(weight, size_average, reduce, reduction_)

    def forward(self, logits, index_of_target, *args, **kwargs):
        """
        Args:
            logits: already softmaxed, NxC
            index_of_target: index of the target [0, C], shape ix N
            *args:
            **kwargs:
        """
        assert len(index_of_target.shape) == 1
        assert len(logits.shape) == 2

        # creates the proba map with one hot encoding
        target = torch.zeros_like(logits)
        target[torch.arange(0, index_of_target.shape[0]), index_of_target] = 1.0
        output = super().forward(logits, target)
        if self.reduction_name == 'mean_per_el':
            output = output / logits.shape[0]
        return output

