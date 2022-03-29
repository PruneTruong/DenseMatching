import torch.utils.data as data
import torch
import numpy as np


class MixDatasets(data.Dataset):
    """Mixes multiple provided datasets. The returned dictionary need to have the same keys and values of
    the same shapes and type. """
    def __init__(self, list_of_datasets, list_overwrite_mask, list_sparse, seed=400):
        """

        Args:
            list_of_datasets:
            list_overwrite_mask: list containing bools, indicating if should overwrite the correspondence mask with
                                 bool array True
            list_sparse:
        """

        self.list_of_datasets = list_of_datasets
        self.list_of_indexes = []
        self.list_overwrite_mask = list_overwrite_mask
        self.list_sparse = list_sparse
        self.get_indexes()

    def get_indexes(self):
        self.list_of_indexes = []
        for index_dataset, dataset in enumerate(self.list_of_datasets):
            for index_element in range(dataset.__len__()):
                self.list_of_indexes.append([index_dataset, index_element])

    def sample_new_items(self, seed):
        self.list_of_indexes = []
        print('Sampling images in mixture')
        for index_dataset, dataset in enumerate(self.list_of_datasets):
            if hasattr(dataset, 'sample_new_items'):
                getattr(dataset, 'sample_new_items')(seed)
            for index_element in range(dataset.__len__()):
                self.list_of_indexes.append([index_dataset, index_element])

    def __getitem__(self, index):
        """
        Args:
            index:

        Returns:
            dict_element, output of __getitem__ of datasets contained in self.list_of_datasets
        """

        [index_dataset, index_element] = self.list_of_indexes[index]
        dict_element = self.list_of_datasets[index_dataset].__getitem__(index_element)

        if self.list_overwrite_mask[index_dataset]:
            if isinstance(dict_element['correspondence_mask'], list):
                list_mask = []
                for i in range(len(dict_element['correspondence_mask'])):
                    if isinstance(dict_element['correspondence_mask'][i], np.ndarray):
                        list_mask.append(np.ones_like(dict_element['correspondence_mask'][i]))
                    else:
                        list_mask.append(torch.ones_like(dict_element['correspondence_mask'][i]))
                dict_element['correspondence_mask'] = list_mask
            else:
                # if it is True, overwrites all to True
                if isinstance(dict_element['correspondence_mask'], np.ndarray):
                    dict_element['correspondence_mask'] = np.ones_like(dict_element['correspondence_mask'])
                else:
                    dict_element['correspondence_mask'] = torch.ones_like(dict_element['correspondence_mask'])
            dict_element['sparse'] = self.list_sparse[index_dataset]

        return dict_element

    def __len__(self):
        """
        total_length = 0
        for dataset in self.list_of_datasets:
            total_length += dataset.__len__()
        """
        total_length = len(self.list_of_indexes)
        return total_length
