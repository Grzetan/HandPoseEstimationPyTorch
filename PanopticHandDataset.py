import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import random
import json

class PanopticHandDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        folders = [os.path.join(path,synth) for synth in os.listdir(path) if '3' in synth]
        self.labels = [os.path.join(folder, f) for folder in folders for f in os.listdir(folder) if '.json' in f]
        self.transforms = transforms
    
    def __len__(self):
        return len(self.labels)

    def get_loader(self, batch_size=32):
        return DataLoader(self, batch_size=batch_size, shuffle=True)

    def __getitem__(self, idx):
        label_path = self.labels[idx]
        p = os.path.normpath(label_path).split(os.path.sep)
        img_path = os.path.join(self.path, p[-2], p[-1].split('.')[0] + '.jpg')
        f = open(label_path, 'r')
        points = np.array(json.load(f)['hand_pts'])[:,:2]
        img = Image.open(img_path).convert('RGB')
        sample = img, points

        if self.transforms is not None:
            sample = self.transforms(sample)
        
        return sample

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    from transforms import *

    transforms = T.Compose([
        Resize()
    ])

    dataset = PanopticHandDataset('./PanopticHandDataset/hand_labels_synth', transforms=transforms)
    for img, points in dataset:
        img = np.asarray(img)
        fig, ax = plt.subplots()
        ax.imshow(img)
        for p in points:
            ax.scatter(p[0], p[1], c='r', s=5)
        plt.show()
    