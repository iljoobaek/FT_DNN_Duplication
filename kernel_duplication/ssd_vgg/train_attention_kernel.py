from data import *
from data.coco import *
from data.voc0712 import *
from data.config import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
# from ssd_subflow_d2nn_1 import build_ssd
from ssd_attention_kernel import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', default=10, type=int,
                    help='Epoch numbers for training')
parser.add_argument('--touch_layer_index', default=1, type=int,
                    help='how many layers to add attention, maximum 3')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=True, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
# parser.add_argument('--trained_model', default='weights/original/VOC.pth',
#                     type=str, help='Trained state_dict file path to open')
parser.add_argument('--trained_model', default='weights/ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--run_original', default=True, type=str2bool,
					help='train the original model')
parser.add_argument('--weight_index', default=1, type=int,
                    help='which layers to pay attention')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

import visdom
viz = visdom.Visdom()

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    ssd_net = build_ssd(args, 'train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    # for name, m in ssd_net.named_parameters():
    #     print(name)
    # exit()

    if not args.run_original:
        net.load_state_dict(torch.load(args.trained_model), strict=False)
        net.attention_mode = True
        net.index = args.weight_index
        net.conv3_attention = nn.Conv2d(net.layer_width[args.weight_index],
                                        net.layer_width[args.weight_index],
                                        3, 1, 1,
                                        groups=net.layer_width[args.weight_index],
                                        bias=False)
        # added_layers = {"fc1.weight", "fc2.weight","fc3.weight", "fc4.weight","fc5.weight", "fc6.weight"}
        # added_layers = {"fc1.weight", "fc2.weight", "fc3.weight", "fc4.weight", "fc5.weight", "fc6.weight",
        #                 "conv1_attention.weight", "conv2_attention.weight", "fc7.weight", "fc8.weight", "mask",
        #                 "conv3_attention.weight"}
        added_layers = {"conv3_attention.weight"}
        for name, param in net.named_parameters():

            if name in added_layers:
                print(name)
                continue
            param.requires_grad = False

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        if args.run_original:
            vgg_weights = torch.load(args.save_folder + args.basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()
        net.conv3_attention = net.conv3_attention.cuda()

    if not args.resume and args.run_original:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    for epoch in range(args.epochs):
        loc_loss = 0
        conf_loss = 0
        for iteration, (images, targets) in enumerate(data_loader):
            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('epoch ' + repr(epoch) + '  ||  iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            if args.visdom:
                update_vis_plot(epoch*epoch_size+iteration, loss_l.item(), loss_c.item(),
                                iter_plot, epoch_plot, 'append')

            if iteration != 0 and iteration % 1000 == 0 and epoch % 2 == 0:
                print('Saving state, iter:', iteration)
                if args.run_original:
                    torch.save(ssd_net.state_dict(), 'weights/original/ssd300_COCO_' +
                               repr(epoch) + '.pth')
                else:
                    torch.save(ssd_net.state_dict(), 'weights/attention3/ssd300_COCO_' +
                               repr(epoch) + '.pth')
    if args.run_original:
        torch.save(ssd_net.state_dict(),
                   args.save_folder + 'original/' + args.dataset + '.pth')
    else:
        torch.save(ssd_net.state_dict(),
                   args.save_folder + 'attention3/' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
