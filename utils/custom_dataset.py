from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class NoiseDataset(Dataset):
    # very redundant, refactor later
    def __init__(self, data, noise_tar, transform=None):
        self.data = data
        self.noise_tar = noise_tar
        self.transform = transform

    def __len__(self):
        return len(self.data.img_labels)

    def __getitem__(self, index):
        img, noise_tar = self.data.get_plain_item(index)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img, noise_tar = self.data.__getitem__(index)
        return img, noise_tar, index


class SplitCandidateDateset(Dataset):
    def __init__(self, data, can_labels, targets, transform=None):
        assert len(can_labels) == len(data)
        self.data = data
        self.can_labels_indexes = np.where(can_labels == 1)
        self.data_indexes = self.can_labels_indexes[0]
        self.can_labels = self.can_labels_indexes[1]
        self.targets = targets
        self.transform = transform
        assert len(self.can_labels) == len(self.data_indexes)

    def __len__(self):
        return len(self.data_indexes)

    def __getitem__(self, index):
        img, target = self.data[self.data_indexes[index]], self.targets[self.data_indexes[index]]
        noise_tar = self.can_labels[index]
        img = Image.fromarray(np.array(img))
        if self.transform is not None:
            img = self.transform(img)
        return img, noise_tar, target, index


class CandidateDateset(Dataset):
    def __init__(self, data, can_labels, targets, transform=None, return_index=False):
        assert len(can_labels) == len(data)
        self.data = data
        self.can_labels = can_labels
        self.targets = targets
        self.transform = transform
        self.return_index = return_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target= self.data.__getitem__(index)
        can_label = self.can_labels[index]
        if self.return_index:
            return img, can_label, target, index
        return img, can_label, target
