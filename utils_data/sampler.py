from torch.utils.data import Sampler
import torch


class RandomSampler(Sampler):
    """Randomly samples items (num_samples) at each epoch. """
    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        new_list = torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist()
        return iter(new_list)

    def __len__(self):
        return self.num_samples


class FirstItemsSampler(Sampler):
    """Samples the first 'num_samples' iterms at each epoch. Useful for degubbing. """
    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        new_list = torch.arange(start=0, end=n, dtype=torch.int64)[: self.num_samples].tolist()
        return iter(new_list)

    def __len__(self):
        return self.num_samples