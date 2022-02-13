import torch
import torchvision.transforms as T
import numpy as np
from random import random
from PIL import Image

class ToTensor(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, sample):
        img, points = sample
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        img = img / 255
        img = torch.tensor(img, dtype=torch.float)
        img = img.permute(2,0,1)
        # img = T.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])(img)
        points = points / self.img_size
        points = torch.tensor(points, dtype=torch.float)
        points = torch.clamp(points, min=0, max=1)
        return img, points

class Resize(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, sample):
        img, points = sample
        original_size = img.size
        r_w = self.img_size / original_size[0]
        r_h = self.img_size / original_size[1] 
        img = img.resize((self.img_size, self.img_size))
        points[:,0] *= r_w
        points[:,1] *= r_h

        return img, points

class RandomVerticalFlip(object):
    def __init__(self, p=0.5, img_size=224):
        self.p = p
        self.trans = T.RandomVerticalFlip(1)
        self.img_size = img_size

    def __call__(self, sample):
        img, points = sample
        if random() < self.p:
            img = self.trans(img)
            points[:,1] = self.img_size - points[:,1]
        return img, points

class RandomColorJitter(object):
    def __init__(self):
        self.trans = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)

    def __call__(self, sample):
        img, points = sample
        img = self.trans(img)
        return img, points        

class RandomPadding(object):
    def __init__(self, max_vertical=100, max_horizontal=100):
        self.max_vertical = max_vertical
        self.max_horizontal = max_horizontal

    def __call__(self, sample):
        img, points = sample

        vertical = int(random() * self.max_vertical)
        horizontal = int(random() * self.max_horizontal)
        top = int(random() * vertical)
        left = int(random() * horizontal)
        width, height = img.size
        new_width = width + horizontal
        new_height = height + vertical
        color = tuple([int(random() * 255) for i in range(3)])
        result = Image.new(img.mode, (new_width, new_height), color)
        result.paste(img, (left, top))
        points[:,0] += left
        points[:,1] += top
        r_w = width / new_width
        r_h = height / new_height
        result = result.resize((width, height))
        points[:,0] *= r_w
        points[:,1] *= r_h
        return result, points
