# -*- coding: utf-8 -*-
# Models/DNNs/lenet.py

import torch
import torch.nn as nn
from Models.DNN import DNN


class LeNet5Backbone(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, input_height=32, input_width=32):
        super(LeNet5Backbone, self).__init__()

        # Conv 1: padding=2 matches TF padding='same'
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv 2: padding=0 matches TF padding='valid'
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        # --- CALCULATE FLATTENED DIMENSION STATICALLY ---
        # Conv 1 output shape: (H + 4 - 5) / 1 + 1 = H
        # Pool 1 output shape: H / 2
        h, w = input_height, input_width

        # After Conv1 & Pool1
        h, w = h // 2, w // 2

        # After Conv2 (no padding): (H - 5) + 1 = H - 4
        h, w = (h - 4), (w - 4)

        # After Pool2
        h, w = h // 2, w // 2

        # Flattened features = Channels (16) * Height * Width
        in_features = 16 * h * w
        # ------------------------------------------------

        # Statically define all linear layers
        self.fc1 = nn.Linear(in_features, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

        self._num_classes = num_classes

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        out = self.relu4(out)
        out = self.fc3(out)
        return out


class Lenet5(DNN):
    name = "Lenet5"
    studentType = ""
    learning_rate_global = 0.001
    learning_rate_local = 0.001
    learningRate_DecayEnable = True

    def __init__(self, num_classes=10, in_channels=3, input_height=32, input_width=32):
        super(Lenet5, self).__init__()
        self.model = LeNet5Backbone(
            num_classes=num_classes,
            in_channels=in_channels,
            input_height=input_height,
            input_width=input_width
        )

    def forward(self, x):
        return self.model(x)
