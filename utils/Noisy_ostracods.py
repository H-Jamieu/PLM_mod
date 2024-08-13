import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Noisy_ostracods(Dataset):
    def __init__(self, root, train, transform=None):

        self.root = root + '/clothing1m'
        self.fixed_annotation_path = f'/mnt/e/HKU_Study/PhD/Noisy_ostracods/datasets/ostracods_genus_final_{train}.csv'
        self.fixed_image_base_path = '/mnt/d/Competetion_data/ostracods_data/class_images'
        self.transform = transform
        self.train = train
        self.img_labels = pd.read_csv(self.fixed_annotation_path, header=None)
        
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)
