import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

class FreiHandDataset(Dataset):
    def __init__(self, imgs_path, xyz, K, n_augmentations=4, transforms=None):
        self.imgs_path = imgs_path
        f_ann = open(xyz, 'r')
        f_k = open(K, 'r')
        self.xyz = np.array(json.load(f_ann))
        self.K = np.array(json.load(f_k))
        self.n_augmentations = n_augmentations
        self.n_images = len(self.xyz) * n_augmentations
        self.transforms = transforms
    
    def __len__(self):
        return self.n_images

    def get_loader(self, batch_size=32):
        return DataLoader(self, batch_size=batch_size)

    def __getitem__(self, idx):
        no_augm_idx = idx%len(self.xyz)
        xyz = self.xyz[no_augm_idx]
        K = self.K[no_augm_idx]
        uv = np.matmul(K, xyz.T).T
        points = uv[:, :2] / uv[:, -1:]
        img_path = self.imgs_path + '/' + '0'*(8 - len(str(idx))) + str(idx) + '.jpg'
        img = Image.open(img_path).convert('RGB')
        sample = (img, points)

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json')
    for img, points in dataset:
        fig, ax = plt.subplots()
        ax.imshow(img)
        for p in points:
            ax.scatter(p[0], p[1], c='r', s=10)
        plt.show()
    