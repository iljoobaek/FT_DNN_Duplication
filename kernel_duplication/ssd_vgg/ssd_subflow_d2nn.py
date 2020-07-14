import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os
import copy


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, args, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.args = args
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.error = None
        self.duplicate_index1 = None
        self.duplicate_index2 = None
        self.duplicate_index3 = None
        self.attention_mode = False
        # self.touch_layers = {1: {2}, 2: {2,5}, 3: {2, 5, 10}}
        self.touch_layers = {1: {22}, 2: {3, 6}, 3: {3, 6, 11}}
        self.weights_copy = {}

        self.output = []
        self.num_duplication = 256
        self.is_importance = False
        self.index_add_output = {1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29, 32, 34}

        # SSD network
        self.vgg = nn.ModuleList(base)
        # 0:3*300 ->(conv) 1:64*300 ->(relu) 2:64*300 ->(conv) 3:64*300 ->(relu) 4:64*300 ->(pool)
        # 5:64*150 ->(conv) 6:128*150 ->(relu) 7:128*150 ->(conv) 8:128*150 ->(relu) 9:128*150 ->(pool)
        # 10:128*75 ->(conv) 11:256*75 ->(relu) 12:256*75 ->(conv) 13:256*75 ->(relu) 14:256*75 ->(conv) 15:256*75 ->(relu) 16:256*75 ->(pool)
        # 17:256*38 ->(conv) 18:512*38 ->(relu) 19:512*38 ->(conv) 20:512*38 ->(relu) 21:512*38 ->(conv) 22:512*38 ->(relu) 23:512*38 ->(pool)
        # 24:512*19 ->(conv) 25:512*19 ->(relu) 26:512*19 ->(conv) 27:512*19 ->(relu) 28:512*19 ->(conv) 29:512*19 ->(relu) 30:512*19 ->(pool)
        # 31:512*19 ->(conv) 32:1024*19 ->(relu) 33:1024*19 ->(conv) 34:1024*19 ->(relu)
        # 0:1024*19 ->(conv+relu) 1:256*19 ->(conv+relu) 2:512*10 ->(conv+relu) 3:128*10 ->(conv+relu) 4:256*5
        # ->(conv+relu) 5:128*5 ->(conv+relu) 6:256*3 ->(conv+relu) 7:128*3

        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.fc1 = nn.Linear(64, 32, bias=False) #32
        self.fc2 = nn.Linear(32, 64, bias=False)

        self.conv1_attention = nn.Conv2d(64, 128, 1, bias=False)  # 32
        self.conv2_attention = nn.Conv2d(128, 64, 1, bias=False)

        self.fc3 = nn.Linear(128, 32, bias=False)
        self.fc4 = nn.Linear(32, 128, bias=False)

        self.fc5 = nn.Linear(256, 32, bias=False)
        self.fc6 = nn.Linear(32, 256, bias=False)

        self.fc7 = nn.Linear(512, 1024, bias=False)
        self.fc8 = nn.Linear(1024, 512, bias=False)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()

    def error_injection(self, x, error_rate, duplicate_index, is_origin, n):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        device = torch.device("cuda")
        origin_shape = x.shape
        if not is_origin:
            total_dim = x[:, :32, :, :].flatten().shape[0]
        else:
            total_dim = x[:, :n, :, :].flatten().shape[0]
            duplicate_index = torch.arange(n).type(torch.long).to(device)
        index = torch.arange(n).type(torch.long).to(device)
        final = torch.stack((duplicate_index, index), axis=0)
        final = final.sort(dim=1)
        reverse_index = final.indices[0]

        x = x[:, duplicate_index, :, :].flatten()
        random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
        m = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([1.0]))
        x[random_index] = m.sample(x[random_index].size()).squeeze()
        # x[random_index] = m.sample(x[random_index].size()).squeeze() - 1 - x[random_index]
        # x[random_index] = 0
        x = x.reshape(origin_shape)
        x = x[:, reverse_index, :, :]

        return x

    def error_injection_1(self, x, error_rate, duplicate_index, is_origin, n):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        device = torch.device("cuda")
        origin_shape = x.shape
        total_dim = x[:, :n, :, :].flatten().shape[0]
        change_dim = x[:, :self.num_duplication, :, :].flatten().shape[0]
        if is_origin:
            duplicate_index = torch.arange(n).type(torch.long).to(device)
        index = torch.arange(n).type(torch.long).to(device)
        final = torch.stack((duplicate_index, index), axis=0)
        final = final.sort(dim=1)
        reverse_index = final.indices[0]

        m = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([1.0]))
        x = x[:, duplicate_index, :, :].flatten()
        x_duplicate = x.clone()
        random_index1 = torch.randperm(total_dim)[:int(total_dim * error_rate)]
        # x[random_index1] = 0
        x[random_index1] = m.sample(x[random_index1].size()).squeeze()
        # x[random_index1] = m.sample(x[random_index1].size()).squeeze() - 1 - x[random_index1]
        if not is_origin:
            random_index2 = torch.randperm(change_dim)[:int(change_dim * error_rate)]
            # x_duplicate[random_index2] = 0
            x_duplicate[random_index2] = m.sample(x[random_index2].size()).squeeze()
            # x_duplicate[random_index2] = m.sample(x[random_index2].size()).squeeze() - 1 - x_duplicate[random_index2]
            x_duplicate[change_dim:total_dim] = x[change_dim:total_dim]
            x = (x+x_duplicate)/2

        x = x.reshape(origin_shape)
        x = x[:, reverse_index, :, :]

        return x

    def error_injection_weights(self, error_rate):
        # error_rate = self.error
        touch1 = {2}
        touch2 = {2, 5}
        touch3 = {2, 5, 10}
        for k in touch1:
            size = self.vgg[k].weight.data.size()
            size1 = self.vgg[k].bias.data.size()
            self.weights_copy[k] = copy.deepcopy(self.vgg[k])
            # print(self.vgg[2].weight.data[0][0])
            total_dim = torch.zeros(size).flatten().shape[0]
            total_dim1 = torch.zeros(size1).flatten().shape[0]
            # print(total_dim)
            random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
            random_index1 = torch.randperm(total_dim1)[:int(total_dim1 * error_rate)]
            x = torch.zeros(total_dim)
            x1 = torch.zeros(total_dim1)
            x[random_index] = 1
            x1[random_index1] = 1
            x = x.reshape(size)
            x1 = x1.reshape(size1)
            m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.5]))
            # print(m.sample(size).size())
            with torch.no_grad():
                self.vgg[k].weight.data = torch.where(x == 0, self.vgg[k].weight.data, m.sample(size).squeeze())
                self.vgg[k].bias.data = torch.where(x1 == 0, self.vgg[k].bias.data, m.sample(size1).squeeze())
                # self.vgg[2].weight.data = torch.where(x == 1, self.vgg[2].weight.data, torch.zeros(size))
            # print(self.vgg[2].weight.data[0][0])

    def error_injection_new(self, x, error_rate):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        origin_shape = x.shape
        total_dim = x.flatten().shape[0]

        x = x.flatten()
        random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]

        m = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([1.0]))
        x[random_index] = m.sample(x[random_index].size()).squeeze()
        # x[random_index] = m.sample(x[random_index].size()).squeeze() - 1 - x[random_index]
        # x[random_index] = 0
        x = x.reshape(origin_shape)

        return x

    def duplication(self, x_original, x_error, duplicate_index):
        x_duplicate = x_error.clone()
        x_duplicate[:, duplicate_index[:self.num_duplication], :, :] = x_original[:, duplicate_index[:self.num_duplication], :, :]
        return x_duplicate

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        if self.is_importance:
            self.output = []
            x = nn.Parameter(x, requires_grad=True)
            x.retain_grad()
            self.output.append(x)

        # apply vgg up to conv4_3 relu
        for k in range(23):

            # x_origin = x.detach().clone()
            # print(k, x.size())
            x = self.vgg[k](x)
            # if k in {2, 5, 10}:
            #     print(self.vgg[k])
            #     tmp = self.vgg[k].weight.data
            #     print(tmp.max(), tmp.min(), tmp.size())
            #     if k == 10:
            #         exit()
            # print(type(self.vgg[k]))
            # if k == 11:
            #     exit()
            # print(x[0][0][0][:100])

            if self.is_importance and k in self.index_add_output:
                x.retain_grad()
                self.output.append(x)

            if self.args.run_original:
                pass
            else:
                if k in self.touch_layers[self.args.touch_layer_index]:
                    if self.error:
                        if self.attention_mode:
                            # if k == 2:
                            #     x_copy = self.weights_copy[k](x_origin)
                            #     # x_dup = self.duplication(x_copy, x, self.duplicate_index1)
                            #     x = (x_copy + x) / 2
                            # if k == 3:
                                # print(x.sum(3).sum(2).sum(0))
                                # x = self.error_injection_1(x, self.error, self.duplicate_index1, is_origin=False, n=64)
                                # print(x.sum(3).sum(2).sum(0))
                                # exit()
                            # elif k == 5:
                            # elif k == 6:
                            #     x = self.error_injection_1(x, self.error, self.duplicate_index2, is_origin=False, n=128)
                            # else:
                            #     x = self.error_injection_1(x, self.error, self.duplicate_index3, is_origin=False, n=256)
                            x_copy = x.clone()
                            # if k == 2:
                            if k == 22:
                                x = self.error_injection_new(x, self.error)
                                x_dup = self.duplication(x_copy, x, self.duplicate_index1)
                            # elif k == 5:
                            elif k == 6:
                                x = self.error_injection_new(x, self.error)
                                x_dup = self.duplication(x_copy, x, self.duplicate_index2)
                            else:
                                x = self.error_injection_new(x, self.error)
                                x_dup = self.duplication(x_copy, x, self.duplicate_index3)
                            x = (x + x_dup) / 2
                        else:
                            # x = x
                            if k == 22:
                                x = self.error_injection(x, self.error, None, is_origin=True, n=512)
                            # if k == 2:
                            # if k == 3:
                            #     # print(x.sum(3).sum(2).sum(0).sum())
                            #     # print(type(self.vgg[k]))
                            #     # print(x[0][0][1])
                            #     x = self.error_injection(x, self.error, None, is_origin=True, n=64)
                            #     # print(x.sum(3).sum(2).sum(0).sum())
                            #     # print(x[0][0][1])
                            #     # exit()
                            # # elif k == 5:
                            # elif k == 6:
                            #     x = self.error_injection(x, self.error, None, is_origin=True, n=128)
                            # else:
                            #     x = self.error_injection(x, self.error, None, is_origin=True, n=256)
                                # exit()
                    else:
                        if self.attention_mode:

                            x = x.permute(0, 2, 3, 1)
                            # if k == 2:
                            # if k == 3:
                            #     # x = self.fc1(x)
                            #     x = self.conv1_attention(x)
                            # print(k)
                            if k == 22:
                                # print("fc7")
                                x = self.fc7(x)

                            # elif k == 5:
                            # elif k == 6:
                            #     x = self.fc3(x)
                            # else:
                            #     x = self.fc5(x)
                            x = nn.Tanh()(x)
                            # if k == 2:
                            # if k == 3:
                            #     # x = self.fc2(x)
                            #     x = self.conv2_attention(x)
                            if k == 22:
                                # print("fc8")
                                x = self.fc8(x)
                            # elif k == 5:
                            # elif k == 6:
                            #     x = self.fc4(x)
                            # else:
                            #     x = self.fc6(x)
                            x = x.permute(0, 3, 1, 2)


        s = self.L2Norm(x)
        sources.append(s)
        #print(len(self.output))

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            # print(k, x.size())
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            # print(k, x.size())
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # exit()
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45, 
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x))                  # default boxes
                # self.priors
            )
            #output = self.detect(
            #    loc.view(loc.size(0), -1, 4),                   # loc preds
            #    self.softmax(conf.view(conf.size(0), -1,
            #                 self.num_classes)),                # conf preds
            #    self.priors.type(type(x))                  # default boxes
                # self.priors
            #)
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []  # localization
    conf_layers = []  # confidence
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        # (4 + classes) * cfg[k]
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(args, phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(args, phase, size, base_, extras_, head_, num_classes)
