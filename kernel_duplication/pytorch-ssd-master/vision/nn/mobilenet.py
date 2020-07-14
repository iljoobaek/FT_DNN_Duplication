# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        # Depthwise Separable Convolution
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),  # in: 224 * 224 * 3
            conv_dw(32, 64, 1),  # in: 112 * 112 * 32
            conv_dw(64, 128, 2),  # in: 112 * 112 * 64
            conv_dw(128, 128, 1),  # in: 56 * 56 * 128
            conv_dw(128, 256, 2),  # in: 56 * 56 * 128
            conv_dw(256, 256, 1),  # in: 28 * 28 * 256
            conv_dw(256, 512, 2),  # in: 28 * 28 * 256
            conv_dw(512, 512, 1),  # in: 14 * 14 * 512
            conv_dw(512, 512, 1),  # in: 14 * 14 * 512
            conv_dw(512, 512, 1),  # in: 14 * 14 * 512
            conv_dw(512, 512, 1),  # in: 14 * 14 * 512
            conv_dw(512, 512, 1),  # in: 14 * 14 * 512
            conv_dw(512, 1024, 2),  # in: 14 * 14 * 512
            conv_dw(1024, 1024, 1),  # in: 7 * 7 * 1024
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x