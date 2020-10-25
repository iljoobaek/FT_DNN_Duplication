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
        self.weights_error = 0
        
        self.recover_type = "KR"

        # importance
        self.output = []
        self.is_importance = False

        # copy weights
        self.weight_index = 0
        self.weights_copy = {}
        self.all_layer_indices = range(1, 13)
        self.width = 0
        self.all_width = {1: 64, 2: 128, 3: 128, 4: 256, 5: 256, 6: 512, 7: 512, 8: 512, 
                         9: 512, 10: 512, 11: 512, 12: 1024} #4924
        self.all_duplication_indices = {}
        self.percentage = 0.5
        # self.percentage_list = [0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.56, 0.2, 0.7, 0.7, 0.27, 0.2]
        # self.percentage_list = [0.1431, 0.1431, 0.1431, 0.1431, 0.1431, 0.1431, 0.1145, 0.0429, 0.1431, 0.1431, 0.0572, 0.0429]
        self.percentage_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        # self.percentage_list = [0.7, 0.5, 0.5, 0.1, 0.04, 0.1, 0.1, 0.2, 0.04, 0.03, 0.025,
        #                         0.03]
        # dup / all = 10
        # self.percentage_list = [0, 0, 0, 0, 0., 0, 0, 0, 0, 0, 0, 0.48]

        # self.percentage_list = [0.5] * 12
        # self.percentage_list = [1] * 12
        self.vgg_all_width = {0: 64, 2: 64, 5: 128, 7: 128, 10: 256, 12: 256, 14: 256, 
                              17: 512, 19: 512, 21: 512, 24: 512, 26: 512, 28: 512, 
                              31: 1024, 33: 1024}
        self.vgg_percentage_list = {0: 0.5, 2: 0.5, 5: 0.5, 7: 0.5, 10: 0.5, 12: 0.5, 14: 0.5, 
                                    17: 0.5, 19: 0.5, 21: 0.5, 24: 0.5, 26: 0.5, 28: 0.5, 
                                    31: 0.5, 33: 0.5}
        
        # entropy
        self.layerwise_entropy = []
        self.entropy_flag = False
        self.softmax = nn.Softmax(dim = 0)
        self.logsoftmax = nn.LogSoftmax(dim = 0)

        self.entropy_flag_p = False
        self.layerwise_entropy_p = []

    def error_injection(self, x, error_rate, duplicate_index, is_origin):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        
        device = torch.device("cuda")
        # device = torch.device("cpu")
        origin_shape = x.shape
        n = origin_shape[1]
        total_dim = x[:, :n, :, :].flatten().shape[0]
        x = x.flatten()
        random_index1 = torch.randperm(total_dim)[:int(total_dim * error_rate)].to(device)
        x[random_index1] = 0

        x = x.reshape(origin_shape)
        return x

    def error_injection_feature(self, x, error_rate, k, x_origin):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        
        device = torch.device("cuda")
        # device = torch.device("cpu")
        origin_shape = x.shape
        n = origin_shape[1]
        total_dim = x[:, :n, :, :].flatten().shape[0]
        x = x.flatten()
        random_index1 = torch.randperm(total_dim)[:int(total_dim * error_rate)].to(device)
        x[random_index1] = 0

        x = x.reshape(origin_shape)
        
        x_dup = x.clone()
        
        start = time.time()
        x_dup[:, self.all_duplication_indices[k][:int(n * self.percentage)], :, :] = \
            (x[:, self.all_duplication_indices[k][:int(n * self.percentage)], :, :] + \
            x_origin[:, self.all_duplication_indices[k][:int(n * self.percentage)], :, :]) / 2
        t = time.time() - start
        return x_dup, t

    def _kernel_error_injection(self, error_rate, unit):
        # device = torch.device("cpu")
        device = self.device
        for module in unit:
            if isinstance(module, nn.Conv2d):
                size = module.weight.data.size()
                total_dim = torch.zeros(size).flatten().shape[0]
                random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
                x = torch.zeros(total_dim)
                x_zero = torch.zeros(size).to(device)
                x[random_index] = 1
                x = x.reshape(size).to(device)
                with torch.no_grad():
                    module.weight.data = torch.where(x == 0, module.weight.data, x_zero)

    def _kernel_recover_and_err(self, k):
        device = self.device
        if isinstance(self.base_net[k], nn.Sequential):
            for i, module in enumerate(self.base_net[k]):
                if isinstance(module, nn.Conv2d):
                    size = module.weight.data.size()
                    total_dim = torch.zeros(size).flatten().shape[0]
                    random_index = torch.randperm(total_dim)[:int(total_dim * self.weights_error)]
                    x = torch.zeros(total_dim)
                    x_zero = torch.zeros(size).to(device)
                    x[random_index] = 1
                    x = x.reshape(size).to(device)
                    with torch.no_grad():
                        module.weight.data = torch.where(x == 0, self.weights_copy[k][i].weight.data, x_zero)
        elif isinstance(self.base_net[k], nn.Conv2d):
            # print(type(self.base_net[k]), k)
            module = self.base_net[k]
            size = module.weight.data.size()
            total_dim = torch.zeros(size).flatten().shape[0]
            random_index = torch.randperm(total_dim)[:int(total_dim * self.weights_error)]
            x = torch.zeros(total_dim)
            x_zero = torch.zeros(size).to(device)
            x[random_index] = 1
            x = x.reshape(size).to(device)
            with torch.no_grad():
                module.weight.data = torch.where(x == 0, self.weights_copy[k].weight.data, x_zero)

    def _recover_weight(self, unit, origin):
        for i, module in enumerate(unit):
            if isinstance(module, nn.Conv2d):
                with torch.no_grad():
                    module.weight.data = copy.deepcopy(origin[i].weight.data)

    def recover(self):
        # print("Recover weight to all layers")
        for k in self.all_layer_indices:
            self._recover_weight(self.base_net[k], self.weights_copy[k])

    def weights_average(self, k):
        for i, module in enumerate(self.base_net[k]):
            if isinstance(module, nn.Conv2d):
                with torch.no_grad():
                    print(self.weights_copy[k][i].weight.data.size())
                    module.weight.data = (self.weights_copy[k][i].weight.data + module.weight.data) / 2
        exit()

    def weights_error_average(self, k):
        device = self.device
        flag = False
        # print(type(self.base_net[k]))
        t = 0
        if isinstance(self.base_net[k], nn.Sequential):
            for i, module in enumerate(self.base_net[k]):
                if isinstance(module, nn.Conv2d):
                    size = module.weight.data.size()
                    total_dim = torch.zeros(size).flatten().shape[0]
                    random_index = torch.randperm(total_dim)[:int(total_dim * self.weights_error)]
                    x = torch.zeros(total_dim)
                    x_zero = torch.zeros(size).to(device)
                    x[random_index] = 1
                    x = x.reshape(size).to(device)
                    random_index1 = torch.randperm(int(total_dim * self.percentage_list[k - 1]))[:int(total_dim * self.percentage_list[k - 1] * self.weights_error)]
                    x1 = torch.zeros(total_dim)
                    # x1[random_index1] = 1
                    x1 = x1.reshape(size).to(device)
                    with torch.no_grad():
                        # err_to_kernel = torch.where(x == 0, module.weight.data, x_zero)
                        err_to_kernel = torch.where(x == 0, self.weights_copy[k][i].weight.data, x_zero)
                        # err_to_origin = torch.where(x1 == 0, self.weights_copy[k][i].weight.data, x_zero)
                        start = time.time()
                        avg_kernel = (self.weights_copy[k][i].weight.data + err_to_kernel) / 2
                        t += time.time() - start
                        # avg_kernel = (err_to_origin + err_to_kernel) / 2
                        if flag:
                            # print(err_to_kernel.size(), self.all_duplication_indices[k][:int(self.all_width[k-1] * self.percentage_list[k-1])].size())
                            # err_to_kernel[self.all_duplication_indices[k][int(self.all_width[k-1] * 0.5):]] = avg_kernel[self.all_duplication_indices[k][int(self.all_width[k-1] * 0.5):]]
                            
                            err_to_kernel[self.all_duplication_indices[k][:int(self.all_width[k] * self.percentage_list[k-1])]] = avg_kernel[self.all_duplication_indices[k][:int(self.all_width[k] * self.percentage_list[k-1])]]
                            module.weight.data = err_to_kernel
                        else:
                            module.weight.data = avg_kernel
                    if not flag:
                        flag = True
        elif isinstance(self.base_net[k], nn.Conv2d):
            module = self.base_net[k]
            size = module.weight.data.size()
            # print(size)
            total_dim = torch.zeros(size).flatten().shape[0]
            random_index = torch.randperm(total_dim)[:int(total_dim * self.weights_error)]
            x = torch.zeros(total_dim)
            x_zero = torch.zeros(size).to(device)
            x[random_index] = 1
            x = x.reshape(size).to(device)
            random_index1 = torch.randperm(int(total_dim * self.vgg_percentage_list[k]))[:int(total_dim * self.vgg_percentage_list[k] * self.weights_error)]
            x1 = torch.zeros(total_dim)
            x1[random_index1] = 1
            x1 = x1.reshape(size).to(device)
            with torch.no_grad():
                # err_to_kernel = torch.where(x == 0, module.weight.data, x_zero)
                err_to_kernel = torch.where(x == 0, self.weights_copy[k].weight.data, x_zero)
                err_to_origin = torch.where(x1 == 0, self.weights_copy[k].weight.data, x_zero)
                # avg_kernel = (self.weights_copy[k][i].weight.data + err_to_kernel) / 2
                start = time.time()
                avg_kernel = (err_to_origin + err_to_kernel) / 2
                t += time.time() - start
                err_to_kernel[self.all_duplication_indices[k][:int(self.vgg_all_width[k] * self.vgg_percentage_list[k])]] = avg_kernel[self.all_duplication_indices[k][:int(self.vgg_all_width[k] * self.vgg_percentage_list[k])]]
                module.weight.data = err_to_kernel
                # print(err_to_kernel.size())
        return t

    def error_injection_weights_all(self, error_rate):
        # print("Inject error to all layers")
        for k in self.all_layer_indices:
            self._kernel_error_injection(error_rate, self.base_net[k])

    def get_layer_width(self):
        length = 0
        for module in self.base_net[self.weight_index]:
            if isinstance(module, nn.BatchNorm2d):
                length = max(module.weight.data.size()[0], length)
        self.width = length
        return length

    # def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        total_time = [0, 0]
        cnt = 0
        # total_time = 0
        terminal_index = 0
        if isinstance(self.source_layer_indexes[0], tuple):
            terminal_index = self.source_layer_indexes[0][0]
        else:
            terminal_index = self.source_layer_indexes[0]
        # print("terminal_index", terminal_index)
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
            for i, layer in enumerate(self.base_net[start_layer_index: end_layer_index]):
                # if not self.run_original and start_layer_index + i == self.weight_index:
                if self.recover_type == "FMR" and (0 == start_layer_index + i or start_layer_index + i > terminal_index or start_layer_index + i not in self.all_layer_indices):
                    # print(i, total_time, "start")
                    start = time.time()
                    x_original = layer(x) # K+FM
                    # total_time += time.time() - start
                    total_time[1] += time.time() - start
                    cnt += 1
                    # print(i, total_time)
                    # print("FMR")
                if not self.run_original and 0 < start_layer_index + i <= terminal_index and start_layer_index + i in self.all_layer_indices:
                    if self.recover_type == "FMR":
                        start = time.time()
                        x_original = self.weights_copy[start_layer_index + i](x) # K+FM
                        # total_time += time.time() - start
                        total_time[1] += time.time() - start
                        cnt += 1
                        # print("FMR")
                        # print(i, total_time)
                    if self.error:
                        # start = time.time()
                        if self.duplicated:
                            # print(start_layer_index + i)
                            if self.recover_type == "KR":
                                t1 = self.weights_error_average(start_layer_index + i)
                                # total_time += t1
                                total_time[1] += t1
                                cnt += 1
                                # print("KR", t1)
                                # print(i, total_time)
                            elif self.recover_type == "FMR":
                                self._kernel_recover_and_err(start_layer_index + i) # K+FM
                        else:
                            # print(start_layer_index + i)
                            self._kernel_recover_and_err(start_layer_index + i)
                        # total_time[1] += time.time() - start
                # exit()

                start = time.time()
                x = layer(x) # original kernel
                # total_time += time.time() - start
                total_time[0] += time.time() - start
                cnt += 1
                # print(i, total_time)
                if not self.run_original and 0 < start_layer_index + i <= terminal_index and start_layer_index + i in self.all_layer_indices:
                    # print((layer.weight.data - self.base_net[start_layer_index: end_layer_index][i].weight.data).sum())
                    if self.error:
                        # start = time.time()
                        if self.duplicated:
                            if self.recover_type == "FMR":
                                x, t2 = self.error_injection_feature(x, self.error, start_layer_index + i, x_original)
                                # total_time += t2
                                total_time[1] += t2
                                cnt += 1
                                # print("FMR")
                                # print(i, total_time)
                            elif self.recover_type == "KR":
                                x = self.error_injection(x, self.error, None, is_origin=True)
                        else:
                            x = self.error_injection(x, self.error, None, is_origin=True)
                        # total_time[0] += time.time() - start
                    elif self.is_importance:
                        x.retain_grad()
                        self.output.append(x)
                    elif self.entropy_flag:
                        feature = x.clone().detach()
                        print(feature.sum(3).sum(2).mean(0))
                        print(feature.sum(3).sum(2).mean(0).size())
                        print(self.softmax(feature.sum(3).sum(2).mean(0)))
                        print(self.logsoftmax(feature.sum(3).sum(2).mean(0)))
                        entropy = -self.softmax(feature.sum(3).sum(2).mean(0)) * self.logsoftmax(feature.sum(3).sum(2).mean(0))
                        print(entropy.size())
                        self.layerwise_entropy.append(entropy.sum())
                    elif self.entropy_flag_p:
                        feature = x.clone().detach()
                        xtemp = torch.flatten(feature.mean(0), start_dim=1)
                        entropy = -self.softmax(xtemp.permute(1, 0)) * self.logsoftmax(xtemp.permute(1, 0))
                        self.layerwise_entropy_p.append(entropy.sum(0))
            
            start = time.time()
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
            # total_time += time.time() - start
            total_time[0] += time.time() - start
            cnt += 1
            # print(total_time)

        start = time.time()
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
        # total_time += time.time() - start
        total_time[0] += time.time() - start
        cnt += 1
        # print(total_time)
        # print(total_time)
        
        if self.is_test:
            start = time.time()
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations.to(self.device), self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            # total_time += time.time() - start
            total_time[0] += time.time() - start
            cnt += 1
            # print(cnt)
            # print(total_time)
            # exit()
            return confidences, boxes, total_time
        else:
            return confidences, locations, total_time

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
