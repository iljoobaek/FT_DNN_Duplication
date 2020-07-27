"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from data import *
from layers.modules import MultiBoxLoss
import torch.utils.data as data
import copy

#from ssd import build_ssd
from ssd_attention_kernel import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
# parser.add_argument('--trained_model',
#                     default='weights/attention3/VOC.pth', type=str,
#                     help='Trained state_dict file path to open')
parser.add_argument('--trained_model',
                    default='weights/attention3/VOC.pth', type=str,
                    help='Trained state_dict file path to open')
# parser.add_argument('--trained_model',
#                     default='weights/attention3/ssd300_COCO_48.pth', type=str,
#                     help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--run_original', default=False, type=str2bool,
					help='train the original model')
parser.add_argument('--attention_mode', default=False, type=str2bool,
					help='train the original model')
parser.add_argument('--error_rate', type=float, default=0, metavar='M',
                        help='error_rate')
parser.add_argument('--touch_layer_index', default=1, type=int,
                    help='how many layers to add attention, maximum 3')
parser.add_argument('--ft_type', default='attention', type=str,
                    help='Type of kernel duplication')
parser.add_argument('--weight_index', default=1, type=int,
                    help='which layers to duplicate')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
           filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    with open("results", 'a') as f:
        f.write(str(args.error_rate) + "|" + str(args.attention_mode) + "|" + args.ft_type + "|" + str(np.mean(aps)) + "\n")
    # print('~~~~~~~~')
    # print('Results:')
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # print('--------------------------------------------------------------')
    # print('Results computed with the **unofficial** Python eval code.')
    # print('Results should be very close to the official MATLAB eval code.')
    # print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        if i % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


# SubFlow calculate importance
def cal_importance(model, train_data):
    """
    s = o^2 * H

    :param model:
    :param train_data:
    :return:
    """
    importance_list = []
    cfg = voc
    for batch_idx, (data, targets) in enumerate(train_data):
        if args.cuda:
            data = Variable(data.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            data = Variable(data)
            targets = [Variable(ann, volatile=True) for ann in targets]
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
        # forward pass
        output = model(data)
        loss_l, loss_c = criterion(output, targets)
        loss = loss_l + loss_c
        loss.backward()
        #loss = F.cross_entropy(output, target)
        #loss.backward()

        for i, out in enumerate(model.output):
            hessian_approximate = out.grad ** 2  # grad^2 -> hessian
            importance = hessian_approximate * (out.detach().clone() ** 2)
            importance_list.append(importance.mean(dim=0))
            #print(importance_list[-1].size())

        break

    return importance_list


# D2NN: weight sum evaluation
def weight_sum_eval(model):
    weights = model.state_dict()
    evaluation = []
    names = []
    # need to find the connection between conv and fc
    for name, m in model.named_modules():
        # print(name)
        # print(name, weights[name + '.weight'].size())
        if isinstance(m, nn.Conv2d):
            # names.append(name)

            # print(name, weights[name + '.weight'].size())
            # output input H W
            if name == 'vgg.' + str(model.layer_width[args.weight_index + 1]):
                evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=3).sum(dim=2).sum(dim=0))
        # elif isinstance(m, nn.Linear):
        #     names.append(name)
            # print(name, weights[name + '.weight'].size())
            # output input
            # evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=0))
        # elif isinstance(m, nn.BatchNorm2d):
        #     print(name)
        #if evaluation: print(names[-1], type(m), evaluation[-1].size())
    # for i in range(len(names)):
    #     print(names[i], evaluation[i].size())
    # print(names[10], evaluation[10].size())
    return evaluation, names


if __name__ == '__main__':
    # load net
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd(args, 'test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model), strict=False)
    net.index = args.weight_index
    net.num_duplication = int(net.layer_width[args.weight_index] * 0.5)

    # num_layer_mp = {1: 64, 2: 128, 3:256}
    num_layer_mp = {1: net.layer_width[args.weight_index], 2: 128, 3: 256}
    # layer_id = {1: 2, 2: 3, 3: 5}
    # layer_id = {1: 10, 2: 3, 3: 5}
    # layer_mp = {1: net.fc1.weight, 2: net.fc3.weight, 3: net.fc5.weight}
    # layer_mp = {1: net.fc7.weight, 2: net.fc3.weight, 3: net.fc5.weight}
    # layer_mp = {1: net.conv3_attention.weight, 2: net.fc3.weight, 3: net.fc5.weight}
    layer_mp = {1: net.conv3_attention.weight}
    # index_mp = {1: net.duplicate_index1, 2: net.duplicate_index2, 3: net.duplicate_index3}

    # load data
    dataset = VOCDetection(args.voc_root, [('2007', set_type)],
                           BaseTransform(300, dataset_mean),
                           VOCAnnotationTransform())
    # for name, p in net.named_parameters():
    #     print(name)
    # net.error_injection_weights(0.05)
    if not args.attention_mode:
        print("Evaluating model without duplication...")
        # net.attention_mode = True
    else:
        print("Evaluating model with duplication...")
        # Change to evaluate the model with attention
        device = torch.device("cuda")

        data_loader = data.DataLoader(dataset, 16,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

        if args.ft_type == "attention":
            print("attention:")
            for k in {1}:
                index = torch.arange(num_layer_mp[k]).type(torch.float).to(device)
                # print(layer_mp[k].size())
                # tmp = net.mask
                # print((tmp - tmp.min()) / (tmp - tmp.min()).sum() * 256)
                # exit()
                # tmp = torch.sum(layer_mp[k], axis=0)
                tmp = layer_mp[k].sum(3).sum(2).sum(1)
                # print(layer_mp[k].shape)
                final = torch.stack((tmp, index), axis=0)
                final = final.sort(dim=1, descending=True)
                if k == 1:
                    net.duplicate_index1 = final.indices[0]
                    # net.duplicate_index1 = torch.tensor([10,30,19,0,55,59,47,29,62,13,20,35,53,37,28,17,26,33,50,5,57,40,32,34,41,18,12,58,45,52,63,56,27,44,16,6,4,43,60,49,46,48,8,2,1,11,21,36,42,24,31,25,51,3,38,54,15,61,7,9,22,39,23,14])
                    print(net.duplicate_index1)
                elif k == 2:
                    net.duplicate_index2 = final.indices[0]
                if k == 3:
                    net.duplicate_index3 = final.indices[0]
        elif args.ft_type == "importance":
            print("importance:")
            for k in {1}:
                index = torch.arange(num_layer_mp[k]).type(torch.float).to(device)
                net_imp = build_ssd(args, 'train', 300, num_classes)
                weights_imp = copy.deepcopy(net.state_dict())
                net_imp.load_state_dict(weights_imp)
                net_imp.index = args.weight_index
                net_imp.is_importance = True
                importance = cal_importance(net_imp, data_loader)
                net_imp.is_importance = False
                # tmp = importance[layer_id[k]].sum(2).sum(1)
                tmp = importance[0].sum(2).sum(1)
                # print(layer_mp[k].shape)
                final = torch.stack((tmp, index), axis=0)
                final = final.sort(dim=1, descending=True)
                if k == 1:
                    net.duplicate_index1 = final.indices[0]
                    print(net.duplicate_index1)
                elif k == 2:
                    net.duplicate_index2 = final.indices[0]
                if k == 3:
                    net.duplicate_index3 = final.indices[0]
        elif args.ft_type == "d2nn":
            print("d2nn:")
            for k in {1}:
                index = torch.arange(num_layer_mp[k]).type(torch.float).to(device)
                weight_sum, _ = weight_sum_eval(net)
                # tmp = torch.sum(layer_mp[k], axis=0)
                # tmp = weight_sum[layer_id[k]]
                tmp = weight_sum[0]
                # print(layer_mp[k].shape, tmp.size())
                final = torch.stack((tmp, index), axis=0)
                final = final.sort(dim=1, descending=True)
                if k == 1:
                    net.duplicate_index1 = final.indices[0]
                    print(net.duplicate_index1)
                elif k == 2:
                    net.duplicate_index2 = final.indices[0]
                if k == 3:
                    net.duplicate_index3 = final.indices[0]
        else:
            print("random")
            for k in {1}:
                index = torch.arange(num_layer_mp[k]).type(torch.float).to(device)
                tmp = torch.randn(num_layer_mp[k]).to(device)
                final = torch.stack((tmp, index), axis=0)
                final = final.sort(dim=1, descending=True)
                if k == 1:
                    net.duplicate_index1 = final.indices[0]
                    print(net.duplicate_index1)
                elif k == 2:
                    net.duplicate_index2 = final.indices[0]
                if k == 3:
                    net.duplicate_index3 = final.indices[0]

        net.duplicated = True
        # net.attention_mode = True

    net.error = args.error_rate
    net.eval()
    print('Finished loading model!')
    # load data
    # dataset = VOCDetection(args.voc_root, [('2007', set_type)],
    #                        BaseTransform(300, dataset_mean),
    #                        VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)
