import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, layer, n_layers, growth_rate):
        super(DenseBlock, self).__init__()

class TransitionLayer(nn.Module):
    def __init__(self, out_filters):
        super(TransitionLayer, self).__init__()
        self.out_filters = out_filters

class InvertedBottleneckLayer(nn.Module):
    def __init__(self):
        super(InvertedBottleneckLayer, self).__init__()

class AAInvertedBottleneckLayer(nn.Module):
    def __init__(self):
        super(AAInvertedBottleneckLayer, self).__init__()

class HandPoseEstimator(nn.Module):
    def __init__(self, img_size=224, growth_rate=10):
        super(HandPoseEstimator, self).__init__()
        # Architecture
        self.stem_1 = DenseBlock(InvertedBottleneckLayer, n_layers=8, growth_rate=growth_rate)
        self.transition_1 = TransitionLayer(out_filters=64) 

        self.stem_2 = DenseBlock(InvertedBottleneckLayer, n_layers=8, growth_rate=growth_rate)
        self.transition_2 = TransitionLayer(out_filters=64)

        self.tail_1 = DenseBlock(AAInvertedBottleneckLayer, n_layers=6, growth_rate=growth_rate)
        self.transition_3 = TransitionLayer(out_filters=64)

        self.tail_2 = DenseBlock(AAInvertedBottleneckLayer, n_layers=8, growth_rate=growth_rate)
        self.transition_4 = TransitionLayer(out_filters=64)

        self.tail_3 = DenseBlock(AAInvertedBottleneckLayer, n_layers=10, growth_rate=growth_rate)
        self.transition_5 = TransitionLayer(out_filters=64)

        self.tail_4 = DenseBlock(AAInvertedBottleneckLayer, n_layers=12, growth_rate=growth_rate)
        self.transition_6 = TransitionLayer(out_filters=128)

        self.tail_5 = DenseBlock(AAInvertedBottleneckLayer, n_layers=14, growth_rate=growth_rate)
        self.transition_7 = TransitionLayer(out_filters=128)

        self.tail_6 = DenseBlock(AAInvertedBottleneckLayer, n_layers=32, growth_rate=growth_rate)

        self.tail_7 = AAInvertedBottleneckLayer()
        self.avgPooling = nn.AvgPool2d(2, stride=2)
        self.out = nn.Conv2d(1, 42, kernel_size=1)

        self.blocks = nn.ModuleList([
            self.stem_1, self.transition_1, self.stem_2, self.transition_2, self.tail_1, self.transition_3,
            self.tail_2, self.transition_4, self.tail_3, self.transition_5, self.tail_4, self.transition_6,
            self.tail_5, self.transition_7, self.tail_6, self.tail_7, self.avgPooling, self.out
        ])

        print(self.blocks)
    
if __name__ == '__main__':
    model = HandPoseEstimator()



