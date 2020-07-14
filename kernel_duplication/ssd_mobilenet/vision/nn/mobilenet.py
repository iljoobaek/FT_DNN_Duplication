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
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        # self.conv1 = conv_bn(3, 32, 2)
        # self.conv2 = conv_dw(32, 64, 1)
        # self.conv3 = conv_dw(64, 128, 2),
        # self.conv4 = conv_dw(128, 128, 1),
        # self.conv5 = conv_dw(128, 256, 2),
        # self.conv6 = conv_dw(256, 256, 1),
        # self.conv7 = conv_dw(256, 512, 2),
        # self.conv8 = conv_dw(512, 512, 1),
        # self.conv9 = conv_dw(512, 512, 1),
        # self.conv10 = conv_dw(512, 512, 1),
        # self.conv11 = conv_dw(512, 512, 1),
        # self.conv12 = conv_dw(512, 512, 1),
        # self.conv13 = conv_dw(512, 1024, 2),
        # self.conv14 = conv_dw(1024, 1024, 1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        # x = self.conv1(x)
        # print(x.size())
        # x = self.conv2(x)
        # print(x.size())
        # x = self.conv3(x)
        # print(x.size())
        # x = self.conv4(x)
        # print(x.size())
        # x = self.conv5(x)
        # print(x.size())
        # x = self.conv6(x)
        # print(x.size())
        # x = self.conv7(x)
        # print(x.size())
        # x = self.conv8(x)
        # print(x.size())
        # x = self.conv9(x)
        # print(x.size())
        # x = self.conv10(x)
        # print(x.size())
        # x = self.conv12(x)
        # print(x.size())
        # x = self.conv13(x)
        # print(x.size())
        # x = self.conv14(x)
        # print(x.size())
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        exit()
        return x