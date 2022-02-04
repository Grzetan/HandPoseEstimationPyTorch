import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import architecture
import numpy as np
import sys

class DenseBlock(nn.Module):
    def __init__(self, in_channels, layer, n_layers, kernel_size=3, growth_rate=10, n_heads=None):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for n in range(n_layers):
            self.layers.append(layer(in_channels + n*growth_rate, growth_rate, kernel_size=kernel_size, n_heads=n_heads))

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
    def __init__(self, in_channels, out_channels, kernel_size=3, e=4, n_heads=None):
        super(InvertedBottleneckLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_in = nn.Conv2d(in_channels, out_channels*e, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels*e)
        pad = kernel_size // 2
        self.depthwise_conv = nn.Conv2d(out_channels*e, out_channels*e, kernel_size=kernel_size, groups=out_channels*e, padding=pad)
        self.conv_out = nn.Conv2d(out_channels*e, out_channels, kernel_size=1)
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
    def __init__(self, in_channels, out_channels, kernel_size, n_heads=4, dv=0.1, dk=0.1, e=4):
        super(AAInvertedBottleneckLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Regular Inverted Bottleneck Layer
        self.conv_in = nn.Conv2d(in_channels, out_channels*e, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels*e)
        pad = kernel_size // 2
        self.depthwise_conv = nn.Conv2d(out_channels*e, out_channels*e, kernel_size=kernel_size, groups=out_channels*e, padding=pad)
        self.conv_out = nn.Conv2d(out_channels*e, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()
        # Attention Augmentation components
        self.n_heads = n_heads
        self.dv = int(dv * out_channels * e)
        self.dk = int(dk * out_channels * e)
        self.aug_conv_out = nn.Conv2d(out_channels*e, out_channels*e-self.dv, kernel_size)
        self.qkv_conv = nn.Conv2d(out_channels*e, 2*self.dk+self.dv, 1) 
        # self.attention_out = nn.Conv2d(cos, self.dv, 1)
        self.AA = AttentionAugmentation2d(2*self.dk+self.dv, self.dk, self.dv, n_heads)

    def forward(self, X):
        X = self.conv_in(X)
        X = self.bn1(X)
        X = self.mish(X)

        X = self.depthwise_conv(X)
        X = self.bn1(X)
        X = self.mish(X)
        # Attention Augmentation
        a = self.aug_conv_out(X)
        attn_out = self.qkv_conv(X)
        attn_out = self.AA(attn_out)
        attn_out = self.attention_out(attention_out)
        attn_out = torch.cat((a, attn_out), dim=1)
        attn_out = self.bn1(attn_out)
        attn_out = self.mish(attn_out)
        X = X + attn_out # Add results of depthwise convolution and AA block
        # Head
        X = self.conv_out(X)
        X = self.bn2(X)
        X = self.mish(X)

    def __repr__(self):
        return f'AAInvertedBottleneckLayer(in_channels={self.in_channels}, out_channels={self.out_channels})'

class AttentionAugmentation2d(nn.Module):
    def __init__(self, in_channels, dk, dv, n_heads=4):
        super(AttentionAugmentation2d, self).__init__()
        self.in_channels = in_channels
        self.dk = dk
        self.dv = dv
        self.n_heads = n_heads
        self.dk_per_head = (self.dk // n_heads) ** -0.5
        self.w = nn.Parameter(data = torch.tensor([1.,1.,1.]))

    def forward(self, X):
        # Split input along channels dim into Keys, Values and Queries
        q, k ,v = torch.split(X, [self.dk, self.dk, self.dv], dim=1)
        # Split Keys, Values and Queries into n_heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        q = q * self.dk_per_head

        # Flatten spatial dimentions
        flat_q = self.flatten_spatial(q)
        flat_k = self.flatten_spatial(k)
        flat_v = self.flatten_spatial(v)

        # Calculate logits        
        logits = torch.matmul(flat_k.transpose(3,2), flat_q)
        print(logits.shape)

        return X

    def split_heads(self, x):
        batch, channels, w, h = x.shape
        x = x.view(batch, self.n_heads, channels // self.n_heads, w, h)
        return x

    def flatten_spatial(self, x):
        batch, n_heads, channels, w, h = x.shape
        x = x.view(batch, n_heads, channels, w*h)
        return x

class HandPoseEstimator(nn.Module):
    def __init__(self, architecture, img_size=224, growth_rate=10):
        super(HandPoseEstimator, self).__init__()
        self.blocks = nn.ModuleList()
        # Architecture
        prev_channels = 3
        for block in architecture:
            if block['type'] == 'Dense':
                if block['layer'] == 'IBL':
                    self.blocks.append(DenseBlock(prev_channels, 
                                                  InvertedBottleneckLayer, 
                                                  block['n_repeats'], 
                                                  block['kernel_size'], 
                                                  growth_rate))
                elif block['layer'] == 'AAIBL':
                    self.blocks.append(DenseBlock(prev_channels, 
                                                  AAInvertedBottleneckLayer, 
                                                  block['n_repeats'], 
                                                  block['kernel_size'],
                                                  growth_rate,
                                                  block['n_heads']))
                prev_channels += growth_rate * block['n_repeats']
            elif block['type'] == 'Transition':
                self.blocks.append(TransitionLayer(prev_channels, block['out_channels']))
                prev_channels = block['out_channels']
            elif block['type'] == 'AAIBL':
                self.blocks.append(AAInvertedBottleneckLayer(prev_channels, block['out_channels'], block['kernel_size']))
            elif block['type'] == 'AvgPool':
                self.blocks.append(nn.AvgPool2d(block['kernel_size'], block['stride']))
            elif block['type'] == 'out':
                self.blocks.append(nn.Conv2d(prev_channels, block['out_channels'], block['kernel_size']))

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
    # for p in model.parameters():
    #     p.requires_grad = True
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

