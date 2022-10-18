import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data_utils
from pathlib import Path

dataset_dir = Path(__file__).parents[1] / 'data/point_process'


def list_datasets():
    check = lambda x: x.is_file() and x.suffix == '.npz'
    file_list = sorted([x.stem for x in (dataset_dir).iterdir() if check(x)])
    return file_list

def load_dataset(name):
    if not name.endswith('.npz'):
        name += '.npz'
    loader = dict(np.load(dataset_dir / name, allow_pickle=True))
    data = loader['data']
    labels = loader.get('labels', np.zeros(len(data)))
    num_classes = len(np.unique(labels))
    return SetDataset(data, labels, num_classes)

def collate(batch):
    """
    Returns:
        data: [batch, num_points, dim]
        mask: [batch, num_points, dim]
        label: [batch]
    """

    labels = torch.Tensor([x[1] for x in batch])

    lens = np.array([len(x[0]) for x in batch])
    dim = batch[0][0].shape[1]
    max_len = max(lens)

    data = [torch.Tensor(x[0]) for x in batch]
    data = torch.stack([F.pad(x, (0, 0, 0, max_len - l)) for x, l in zip(data, lens)])

    mask = (torch.arange(max_len).expand(len(lens), max_len) < torch.Tensor(lens)[:,None]).float()

    return data, mask.unsqueeze(-1)

def collate_sort(batch):
    """ Returns same as collate func but sorted on first column of last dim. """

    labels = torch.Tensor([x[1] for x in batch])

    lens = np.array([len(x[0]) for x in batch])
    dim = batch[0][0].shape[1]
    max_len = max(lens)

    data = [torch.Tensor(x[0][x[0][:,0].argsort()]) for x in batch]
    data = torch.stack([F.pad(x, (0, 0, 0, max_len - l), value=1) for x, l in zip(data, lens)])

    mask = (torch.arange(max_len).expand(len(lens), max_len) < torch.Tensor(lens)[:,None]).float()

    return data, mask.unsqueeze(-1)


class SetDataset(data_utils.Dataset):
    def __init__(self, data, labels, num_classes):
        self.num_classes = num_classes
        self.data = data
        self.labels = labels

    def split_train_val_test(self, train_size=0.6, val_size=0.2):
        ind1 = int(len(self.data) * train_size)
        ind2 = ind1 + int(len(self.data) * val_size)

        trainset = SetDataset(self.data[:ind1], self.labels[:ind1], self.num_classes)
        valset = SetDataset(self.data[ind1:ind2], self.labels[ind1:ind2], self.num_classes)
        testset = SetDataset(self.data[ind2:], self.labels[ind2:], self.num_classes)

        return trainset, valset, testset

    @property
    def dim(self):
        return self.data[0].shape[-1]

    def __getitem__(self, key):
        return self.data[key], self.labels[key]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'SetDataset({self.__len__()})'
