import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Noisy_ostracods(Dataset):
    def __init__(self, root, train, transform=None):

        self.root = root + '/clothing1m'
        self.fixed_annotation_path = f'/mnt/e/HKU_Study/PhD/Noisy_ostracods/datasets/ostracods_genus_final_{train}.csv'
        self.fixed_image_base_path = '/mnt/x/class_images' #'/mnt/e/data/ostracods_id/class_images'
        self.transform = transform
        self.train = train
        self.img_labels = pd.read_csv(self.fixed_annotation_path, header=None)
        # transform img_labels[,1] to a 1-d array of np-int8 as labels
        self.targets = self.img_labels[1].values.astype(np.int8)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.fixed_image_base_path, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = self.transform(image)
        label = self.img_labels.iloc[idx, 1]
        return image, label
    
    def get_plain_item(self, idx):
        img_path = os.path.join(self.fixed_image_base_path, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        return image, label

    def __len__(self):
        return len(self.img_labels)
