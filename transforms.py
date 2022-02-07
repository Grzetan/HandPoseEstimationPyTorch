import torch
import torchvision.transforms as transforms
import numpy as np

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
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(img)
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
        