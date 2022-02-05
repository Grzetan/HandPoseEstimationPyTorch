import torch

class ToTensor(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, sample):
        img, points = sample
        img = img / 255
        img = torch.tensor(img, dtype=torch.float16)
        img = img.permute(2,0,1)
        points = points / self.img_size
        points = torch.tensor(points, dtype=torch.float16)
        return img, points