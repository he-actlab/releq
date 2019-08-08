#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['lenet_mnist']

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(800, 500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)
        self.relu4 = nn.ReLU()
        self.activations = {}

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #self.act_conv2 = self.pool(F.relu(self.conv1(x)))
        #self.act_conv2 = F.relu(self.conv2(self.act_conv2))
        x = self.pool(self.relu1(self.conv1(x)))
        #print(set(x.data.cpu().numpy().ravel()))
        x = self.pool(self.relu2(self.conv2(x)))
        self.act_conv2 = x
        torch.set_printoptions(precision=10)
        x = x.view(-1, 800)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        self.act_fc2 = x
        #x = nn.Threshold(0.2, 0.0)#ActivationZeroThreshold(x)
        return x
    
    def freeze(self):
        child_counter = 0
        for child in self.children():
            for param in child.parameters():
                param.requires_grad = False
            child_counter += 1
    
    def freeze_partial(self, layer_list):
        child_counter = 0
        for child in self.children():
            if child_counter not in layer_list:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
            child_counter += 1

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def lenet_mnist():
    model = Lenet()
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.activations = activation
    return model
