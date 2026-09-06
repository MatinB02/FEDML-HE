# -*- coding: utf-8 -*-
# Models/DNNs/resnet.py

import torch
import torch.nn as nn
from Models.DNN import DNN

BN_EPSILON = 1e-5  # PyTorch standard (or 0.001 to strictly match original Keras)
FILTER_CHANNELS = 3

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, eps=BN_EPSILON)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, eps=BN_EPSILON)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

##########################################################################

class ResNetBackbone(nn.Module):
    def __init__(self, layers_cfg, num_classes=10, in_channels=3):
        super(ResNetBackbone, self).__init__()
        self.in_planes = 64

        # CIFAR-style stem: 3x3 conv, no initial maxpool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=BN_EPSILON)
        self.relu = nn.ReLU(inplace=True)

        # ResNet stages
        self.layer1 = self._make_layer(64, layers_cfg[0], stride=1)
        self.layer2 = self._make_layer(128, layers_cfg[1], stride=2)
        self.layer3 = self._make_layer(256, layers_cfg[2], stride=2)
        self.layer4 = self._make_layer(512, layers_cfg[3], stride=2)

        # Global average pooling + classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion, eps=BN_EPSILON)
            )

        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

##########################################################################

class ResNet18(DNN):
    name = "ResNet18"
    studentType = ""
    learning_rate_global = 0.001
    learning_rate_local = 0.001
    learningRate_DecayEnable = True

    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet18, self).__init__()
        self.model = ResNetBackbone(layers_cfg=[2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)

    def forward(self, x):
        return self.model(x)


class ResNet34(DNN):
    name = "ResNet34"
    studentType = ""
    learning_rate_global = 0.001
    learning_rate_local = 0.001
    learningRate_DecayEnable = True

    def __init__(self, num_classes=10, in_channels=3):
        super(ResNet34, self).__init__()
        self.model = ResNetBackbone(layers_cfg=[3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

    def forward(self, x):
        return self.model(x)
