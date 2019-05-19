"""AlexNet"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet with BN and pool5 to be AdaptiveAvgPool2d(1)"""
    def __init__(self, with_classifier=False, return_conv=False, num_classes=1000):
        super(AlexNet, self).__init__()
        self.with_classifier = with_classifier
        self.return_conv = return_conv
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.AdaptiveAvgPool2d(1)

        if self.return_conv:
            self.feature_pool = nn.MaxPool2d(kernel_size=3, stride=2)   # 9216

        if self.with_classifier:
            self.linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        if self.return_conv:
            x = self.feature_pool(x)
            # print(x.shape)
            return x.view(x.shape[0], -1)

        x = self.pool5(x)

        x = x.view(-1, 256)

        if self.with_classifier:
            x = self.linear(x)
        # print(x.shape)
        return x