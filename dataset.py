import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class FreiHandDataset(Dataset):
    def __init__(self, imgs_path, annotations, n_augmentations=4, transforms=None):
        self.imgs_path = imgs_path
        f = open(annotations, 'r')
        self.annotations = np.array(json.load(f))[:,:2] # Only x and y coordinates
        self.n_augmentations = n_augmentations
        self.n_images = len(self.annotations) * n_augmentations
        self.transforms = transforms
    
    def __len__(self):
        return self.n_images

    def get_loader(self, batch_size=32):
        return DataLoader(self, batch_size=batch_size)

    def __getitem__(self, idx):
        points = self.annotations[idx%len(self.annotations)]
        img_path = self.imgs_path + '/' + '0'*(8 - len(str(idx))) + str(idx) + '.jpg'
        img = Image.open(img_path).convert('RGB')


if __name__ == '__main__':
    dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json')
    dataset[1000]
    