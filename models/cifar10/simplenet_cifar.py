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

__all__ = ['simplenet_cifar']

class Simplenet(nn.Module):
    def __init__(self):
        super(Simplenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #self.act_conv2 = self.pool(F.relu(self.conv1(x)))
        #self.act_conv2 = F.relu(self.conv2(self.act_conv2))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        self.act_conv2 = x
        x = F.relu(self.fc2(x))
        #x = nn.Threshold(0.2, 0.0)#ActivationZeroThreshold(x)
        x = self.fc3(x)
        self.act_conv2 = x
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

def simplenet_cifar():
    model = Simplenet()
    return model
