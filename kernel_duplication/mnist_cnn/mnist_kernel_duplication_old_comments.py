from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import os
import time

class SimpleCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=28, c_dim=1, error=0, duplicate_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.error = error
        self.duplicate_index = duplicate_index

        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.nonlinear = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.flat_dim = inp_size*inp_size*4
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))
        self.fc3 = nn.Linear(32, 20, bias=False)
        self.fc4 = nn.Linear(20, 32, bias=False)


    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification score in shape of (N, Nc)
        """
        N = x.size(0)
        x = self.conv1(x)
        x = self.nonlinear(x)


        '''
            Simulate error injection
        '''
        if self.error:

            x_copy = x.clone()
            x = error_injection(x, self.error, self.duplicate_index)  # duplicate_index = [1 0 2]

            # x_copy: feature map without error
            # x: feature map with error
            # duplicate 20: final (15 + 0)/2 = 7.5
            # un-duplicated 12: final 0



            x = (x+x_copy)/2
            pass

        else:
            x = x.permute(0, 2, 3, 1)  # N * C * W * H  ->  N * W * H * C
            x = self.fc3(x)   # from 32 to 20
            x = self.fc4(x)   # from 20 back to 32
            x = x.permute(0, 3, 1, 2)  # N * W * H * C ->  N * C * W * H

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nonlinear(x)
        x = self.pool2(x)

        flat_x = x.view(N, self.flat_dim)
        out = self.fc1(flat_x)
        out = self.fc2(out)
        return out  # N* 10


def get_fc(inp_dim, out_dim, non_linear='relu'):
    """
    Mid-level API. It is useful to customize your own for large code repo.
    :param inp_dim: int, intput dimension
    :param out_dim: int, output dimension
    :param non_linear: str, 'relu', 'softmax'
    :return: list of layers [FC(inp_dim, out_dim), (non linear layer)]
    """
    layers = []
    layers.append(nn.Linear(inp_dim, out_dim))
    if non_linear == 'relu':
        layers.append(nn.ReLU())
    elif non_linear == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif non_linear == 'none':
        pass
    else:
        raise NotImplementedError
    return layers


def error_injection(x, error_rate, duplicate_index):
    """
        Simulate Error Injection.
        :param x: tensor, input tensor (in our case CNN feature map)
        :param error_rate: double, rate of errors
        :return: tensor, modified tensor with error injected
    """
    origin_shape = x.shape  # N * 32 * W * H
    total_dim = x[:, :20, :, :].flatten().shape[0]    # total_dim = N * 20 * W * H
    index = torch.arange(32).type(torch.long).to(device)  # index: 0 1 2... 31
    final = torch.stack((duplicate_index, index), axis=0)
    # 0 1
    # 1 0
    # 2 2
    final = final.sort(dim=1)
    reverse_index = final.indices[0]

    # 0
    # 1
    # 2


    x = x[:,duplicate_index,:,:].flatten()  # 0 1 2 -> 1 0 2

    # 1
    # 0
    # 2

    # 32*10
    # 20*10:
    # 51 2 31... 199
    random_index = torch.randperm(total_dim)[:int(total_dim*error_rate)]

    # Error injection
    x[random_index] = 0
    x = x.reshape(origin_shape)
    # 1  0
    # 0  1
    # 2  2

    x = x[:, reverse_index, :, :]

    return x


def main():
    # 1. load dataset and build dataloader
    model_dir = "../model/RA1/"
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True)

    # 2. define the model, and optimizer.
    PATH = "../model/RA/epoch-4.pt"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(PATH), strict=False)
    added_layers = {"fc3.weight", "fc4.weight"}
    for name, param in model.named_parameters():
        if (name in added_layers):
            continue
        param.requires_grad = False

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    cnt = 0
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get a batch of data
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Calculate the loss
            loss = F.cross_entropy(output, target)
            # Calculate gradient w.r.t the loss
            loss.backward()
            # Optimizer takes one step
            optimizer.step()
            # Log info
            if cnt % args.log_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            # Validation iteration
            if cnt % args.val_every == 0:
                test_loss, test_acc = test(model, device, test_loader)

                model.train()
            cnt += 1
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))


    PATH = "../model/RA1/epoch-4.pt"
    model = SimpleCNN(error=0.95).to(device)
    model.load_state_dict(torch.load(PATH), strict=False)

    index = torch.arange(32).type(torch.float).to(device)  # 1,2,3,4,....32

    # 20 * 32  ->  1 * 32:  10 20 30 ... 100
    # 2*3:
    # 5 7 9
    # 1 2 3
    # 4 5 6

    tmp = torch.sum(model.fc3.weight, axis=0)  # 2 * 3  ->  1 * 3:  10 5 20
    final = torch.stack((tmp, index), axis=0)

    # final: # 0 10    ->   1 5
             # 1 5     ->   0 10
             # 2 20    ->   2 20
    final = final.sort(dim=1)


    model.duplicate_index = final.indices[0]  # 1 0 2



    test_loss, test_acc = test(model, device, test_loader)
    print("final acc: ", test_acc)



def test(model, device, test_loader):
    """Evaluate model on test dataset."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, correct / len(test_loader.dataset)



def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # config for dataset
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--val_every', type=int, default=100, metavar='N',
                        help='how many batches to wait before evaluating model')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device


if __name__ == '__main__':
    args, device = parse_args()
    main()