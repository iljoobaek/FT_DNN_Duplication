import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
import copy
import time

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

        # error injection
        self.error = 0
        self.num_duplication = 5
        self.run_original = False
        self.duplicated = False
        self.attention_mode = False

        # attention
        self.fc1 = nn.Linear(64, 32, bias=False)
        self.fc2 = nn.Linear(32, 64, bias=False)

        self.conv1_attention = nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False)

        # importance
        self.output = []
        self.is_importance = False

        # copy weights
        self.weight_index = 0
        self.weights_copy = {}
        self.all_layer_indices = range(1, 13)
        self.width = 0
        self.all_width = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
        self.all_duplication_indices = {}
        self.percentage = 0.5

    def error_injection(self, x, error_rate, duplicate_index, is_origin, n, x_dup=None):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        # print(x.shape)
        device = torch.device("cuda")
        origin_shape = x.shape
        # total_dim = x.flatten().shape[0]
        total_dim = x[:, :n, :, :].flatten().shape[0]
        # change_dim = x[:, :self.num_duplication, :, :].flatten().shape[0]

        change_dim = x[:, :int(n * self.percentage), :, :].flatten().shape[0]
        if is_origin:
            # random_index1 = torch.randperm(total_dim)[:int(total_dim * error_rate)]
            # x[random_index1] = 0
            # return
            duplicate_index = torch.arange(n).type(torch.long).to(device)
        index = torch.arange(n).type(torch.long).to(device)
        final = torch.stack((duplicate_index, index), axis=0)
        final = final.sort(dim=1)
        reverse_index = final.indices[0]

        m = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([1.0]))
        x = x[:, duplicate_index, :, :].flatten()
        # x = x.flatten()
        # zeromat = torch.zeros(x.size()).to(device)
        random_index1 = torch.randperm(total_dim)[:int(total_dim * error_rate)].to(device)
        # x[random_index1] = zeromat[random_index1]
        # x[random_index1] = 0
        x[random_index1] = m.sample(x[random_index1].size()).squeeze().to(device)

        if x_dup is not None:
            x_duplicate = x_dup[:, duplicate_index, :, :].flatten()
            # random_index2 = torch.randperm(change_dim)[:int(change_dim * error_rate)]
            # x_duplicate[random_index2] = 0
            # x_duplicate[random_index2] = m.sample(x[random_index2].size()).squeeze()
            # x_duplicate[random_index2] = m.sample(x[random_index2].size()).squeeze() - 1 - x_duplicate[random_index2]
            x_duplicate[change_dim:total_dim] = x[change_dim:total_dim]
            x = (x+x_duplicate)/2

        # x[random_index1] = m.sample(x[random_index1].size()).squeeze().to(device)
        # x[random_index1] = m.sample(x[random_index1].size()).squeeze() - 1 - x[random_index1]


        x = x.reshape(origin_shape)
        x = x[:, reverse_index, :, :]

        return x

    def _kernel_error_injection(self, error_rate, unit):
        for module in unit:
            if isinstance(module, nn.Conv2d):
                size = module.weight.data.size()
                total_dim = torch.zeros(size).flatten().shape[0]
                random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
                x = torch.zeros(total_dim)
                x_zero = torch.zeros(size).to(self.device)
                x[random_index] = 1
                x = x.reshape(size).to(self.device)
                with torch.no_grad():
                    module.weight.data = torch.where(x == 0, module.weight.data, x_zero)

    def error_injection_weights_all(self, error_rate):
        print("Inject error to all layers")
        for k in self.all_layer_indices:
            self._kernel_error_injection(error_rate, self.base_net[k])

    def error_injection_weights(self, error_rate):
        # error_rate = self.error
        # length = 0
        touch1 = {self.weight_index}
        for k in touch1:
            # print(type(self.base_net[k]))
            # print(self.base_net[k])
            for module in self.base_net[k]:
                # print(module)
                # if isinstance(module, nn.BatchNorm2d):
                #     length = max(module.weight.data.size()[0], length)
                if isinstance(module, nn.Conv2d):
                    # print(module.weight.data.size())
                    size = module.weight.data.size()
                    # if size[1] == 1:
                    #     length = size[0]
                    # size1 = m.bias.data.size()
                    total_dim = torch.zeros(size).flatten().shape[0]
                    # total_dim1 = torch.zeros(size1).flatten().shape[0]
            # print(total_dim)
                    random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
                    # random_index1 = torch.randperm(total_dim1)[:int(total_dim1 * error_rate)]
                    x = torch.zeros(total_dim)
                    # x1 = torch.zeros(total_dim1)
                    x_zero = torch.zeros(size).to(self.device)
                    x[random_index] = 1
                    # x1[random_index1] = 1
                    x = x.reshape(size).to(self.device)
                    # x1 = x1.reshape(size1)
                    m = torch.distributions.normal.Normal(torch.tensor([0.0]).to(self.device), torch.tensor([0.5]).to(self.device))
            # print(m.sample(size).size())
                    with torch.no_grad():
                        # print(m.sample(size).size(), m.sample(size).squeeze(dim=-1).size())
                        module.weight.data = torch.where(x == 0, module.weight.data, m.sample(size).squeeze(dim=-1))
                        # module.weight.data = torch.where(x == 0, module.weight.data, x_zero)
                        # m.bias.data = torch.where(x1 == 0, m.bias.data, m.sample(size1).squeeze())
                        # self.vgg[2].weight.data = torch.where(x == 1, self.vgg[2].weight.data, torch.zeros(size))
            # print(self.vgg[2].weight.data[0][0])
        # self.width = length
        # return length

    def get_layer_width(self):
        length = 0
        for module in self.base_net[self.weight_index]:
            if isinstance(module, nn.BatchNorm2d):
                length = max(module.weight.data.size()[0], length)
        self.width = length
        return length

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        total_time = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            # for layer in self.base_net[start_layer_index: end_layer_index]:
            #     x = layer(x)
            for i, layer in enumerate(self.base_net[start_layer_index: end_layer_index]):
                if not self.run_original and start_layer_index + i == self.weight_index:
                # if not self.run_original and 0 < start_layer_index + i < 13:

                    # x_copy = copy.deepcopy(x)
                    x_copy = x.detach().clone()
                    # pass

                x = layer(x) # original kernel
                # x_tmp = x.detach().clone()
                # print(start_layer_index + i, x.size())
                if not self.run_original and start_layer_index + i == self.weight_index:
                # if not self.run_original and 0 < start_layer_index + i < 13:
                    if self.error:
                        # print(self.error)
                        if self.duplicated:
                            # x_dup = self.weights_copy[self.weight_index](x_copy) # duplicated kernel
                            x_dup = self.weights_copy[start_layer_index + i](x_copy)
                            # x = self.error_injection(x, self.error, self.duplicate_index1, is_origin=False, n=self.width, x_dup=x_dup)
                            x = self.error_injection(x, self.error, self.all_duplication_indices[start_layer_index + i],
                                                     is_origin=False, n=self.all_width[start_layer_index + i - 1],
                                                     x_dup=x_dup)

                        else:
                            start = time.time()
                            x = self.error_injection(x, self.error, None, is_origin=True, n=self.all_width[start_layer_index + i - 1])
                            total_time += time.time() - start
                    elif self.attention_mode:
                        # print("train attention")
                        # x = x.permute(0, 2, 3, 1)
                        # x = self.fc1(x)
                        # x = nn.Tanh()(x)
                        # x = self.fc2(x)
                        # x = x.permute(0, 3, 1, 2)
                        x = self.conv1_attention(x)
                    elif self.is_importance:
                        x.retain_grad()
                        self.output.append(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        # print(total_time)
        
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations.to(self.device), self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
