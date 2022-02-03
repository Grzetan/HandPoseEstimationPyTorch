import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import architecture
import numpy as np
import sys

class DenseBlock(nn.Module):
    def __init__(self, in_channels, layer, n_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for n in range(n_layers):
            self.layers.append(layer(in_channels + n*growth_rate, growth_rate))

    def forward(self, X):
        for l in self.layers:
            out = l(X)
            X = torch.cat((X, out), 1)
        return X

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(TransitionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        k = torch.tensor([[1.,2.,1.],
                          [2.,4.,2.],
                          [1.,2.,1.]])
        k /= torch.sum(k)
        k = k.view(1,1,3,3).repeat(out_channels,1,1,1)
        self.kernel = nn.Parameter(data=k, requires_grad=True)
        self.padding = nn.ReflectionPad2d([1,1,1,1])

    def forward(self, X):
        X = self.conv(X)
        X = F.conv2d(self.padding(X), self.kernel, stride=self.stride, groups=self.out_channels)
        X = self.bn(X)
        return X

    def __repr__(self):
        return f'TransitionLayer(in_channels={self.in_channels}, out_channels={self.out_channels})'

class InvertedBottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, e=4):
        super(InvertedBottleneckLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_in = nn.Conv2d(in_channels, in_channels*e, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels*e)
        pad = kernel_size // 2
        self.depthwise_conv = nn.Conv2d(in_channels*e, in_channels*e, kernel_size=kernel_size, groups=in_channels*e, padding=pad)
        self.conv_out = nn.Conv2d(in_channels*e, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()

    def forward(self, X):
        X = self.conv_in(X)
        X = self.bn1(X)
        X = self.mish(X)

        X = self.depthwise_conv(X)
        X = self.bn1(X)
        X = self.mish(X)

        X = self.conv_out(X)
        X = self.bn2(X)
        X = self.mish(X)
        
        return X

    def __repr__(self):
        return f'InvertedBottleneckLayer(in_channels={self.in_channels}, out_channels={self.out_channels})'

class AAInvertedBottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads=4, e=4):
        super(AAInvertedBottleneckLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Regular Inverted Bottleneck Layer
        self.conv_in = nn.Conv2d(in_channels, out_channels*e, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels*e)
        self.depthwise_conv = nn.Sequential(*[
            nn.Conv2d(out_channels*e, out_channels*e, kernel_size=3, groups=out_channels*e),
            nn.Conv2d(out_channels*e, out_channels*e, kernel_size=1)
        ])
        self.conv_out = nn.Conv2d(out_channels*e, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()
        # Attention Augmentation components
        self.n_heads = n_heads

    def forward(self, X):
        return X

    def __repr__(self):
        return f'AAInvertedBottleneckLayer(in_channels={self.in_channels}, out_channels={self.out_channels})'

class HandPoseEstimator(nn.Module):
    def __init__(self, architecture, img_size=224, growth_rate=10):
        super(HandPoseEstimator, self).__init__()
        self.blocks = nn.ModuleList()
        # Architecture
        prev_channels = 3
        for block in architecture:
            if block[0] == 'Dense':
                if block[1] == 'IBL':
                    self.blocks.append(DenseBlock(prev_channels, InvertedBottleneckLayer, block[2], growth_rate))
                elif block[1] == 'AAIBL':
                    self.blocks.append(DenseBlock(prev_channels, AAInvertedBottleneckLayer, block[2], growth_rate))
                prev_channels += growth_rate * block[2]
            elif block[0] == 'Transition':
                self.blocks.append(TransitionLayer(prev_channels, block[1]))
                prev_channels = block[1]
            elif block[0] == 'AAIBL':
                self.blocks.append(AAInvertedBottleneckLayer(prev_channels))
            elif block[0] == 'AvgPool':
                self.blocks.append(nn.AvgPool2d(block[1], block[2]))
            elif block[0] == 'out':
                self.blocks.append(nn.Conv2d(prev_channels, block[1], kernel_size=1))

    def forward(self, X):
        for block in self.blocks:
            X = block(X)

        return X

if __name__ == '__main__':
    import torchvision.transforms as T
    from transforms import *
    from dataset import FreiHandDataset
    import time

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transforms = T.Compose([
        ToTensor()
    ])
    dataset = FreiHandDataset('./FreiHand/training/rgb', './FreiHand/training_xyz.json', './FreiHand/training_K.json', transforms=transforms)
    loader = dataset.get_loader(batch_size=1)
    model = HandPoseEstimator(architecture)
    for p in model.parameters():
        p.requires_grad = True
    model.half()
    model.to(device)

    for imgs, points in loader:
        imgs = imgs.to(device)
        start = time.time()
        print('Before forward', torch.cuda.memory_allocated(0))
        output = model(imgs)
        print('After forward', torch.cuda.memory_allocated(0)) 
        print(time.time() - start)
        break

