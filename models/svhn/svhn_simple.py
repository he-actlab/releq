import torch.nn as nn
import math
import itertools
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

__all__ = ['svhn']

model_urls = {
    'svhn': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/svhn-f564f3d8.pth',
}


class SVHN(nn.Module):
    def __init__(self, features, num_classes=10):
        super(SVHN, self).__init__()
	    #
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(256, num_classes))
        #
    def forward(self, x):
        self.act_conv2 = self.features[0:7](x)
        x = self.features(x)
        # size(0) is batch size
        # size(1) is image size
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def freeze(self):
        child_counter = 0
        for child in itertools.chain(self.features, self.classifier):
            for param in child.parameters():
                param.requires_grad = False
            child_counter += 1
    
    def freeze_partial(self, layer_list):
        child_counter = 0
        for child in itertools.chain(self.features, self.classifier):
            if child_counter not in layer_list:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)

n_channel = 32
cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']

def svhn(**kwargs):
    model = SVHN(make_layers(cfg), **kwargs)
    return model