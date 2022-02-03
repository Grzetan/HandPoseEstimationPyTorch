import torch

class ToTensor(object):
    def __call__(self, sample):
        img, points = sample
        img = img / 255
        img = torch.tensor(img, dtype=torch.float16)
        img = img.permute(2,0,1)
        points = torch.tensor(points)
        return img, points