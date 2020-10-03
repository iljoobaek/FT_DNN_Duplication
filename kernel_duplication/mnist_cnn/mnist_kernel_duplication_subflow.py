from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
# import cudnn
# import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter
import os
import time
import copy

class SimpleCNN(nn.Module):
    """
    Model definition
    """
    def __init__(self, num_classes=10, inp_size=28, c_dim=1, error=0, num_duplication=5, duplicate_index=None, attention_mode=False, error_spread=False):
        super().__init__()
        self.num_classes = num_classes
        self.error = error
        self.duplicate_index = duplicate_index

        self.output = []

        self.conv1 = nn.Conv2d(c_dim, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)

        self.nonlinear = nn.ReLU()
        self.pool1 = nn.AvgPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.attention_mode = attention_mode
        self.evaluate_origin = False
        self.num_duplication = num_duplication
        self.error_spread = error_spread
        
        self.importance = False
        self.entropy = False
        self.layerwise_entropy_p = []
        self.softmax = nn.Softmax(dim = 0)
        self.logsoftmax = nn.LogSoftmax(dim = 0)

        self.flat_dim = inp_size*inp_size*4
        self.fc1 = nn.Sequential(*get_fc(self.flat_dim, 128, 'relu'))
        self.fc2 = nn.Sequential(*get_fc(128, num_classes, 'none'))
        self.fc3 = nn.Linear(32, 20, bias=False) # relate to num_duplication? no
        self.fc4 = nn.Linear(20, 32, bias=False)

    def error_injection(self, x, error_rate, duplicate_index, is_origin):
        """
            Simulate Error Injection.
            :param x: tensor, input tensor (in our case CNN feature map)
            :param error_rate: double, rate of errors
            :return: tensor, modified tensor with error injected
        """
        origin_shape = x.shape
        if not is_origin:
            total_dim = x[:, :self.num_duplication, :, :].flatten().shape[0]
        else:
            total_dim = x[:, :32, :, :].flatten().shape[0]
            duplicate_index = torch.arange(32).type(torch.long).to(device)
        index = torch.arange(32).type(torch.long).to(device)
        final = torch.stack((duplicate_index, index), dim=0)
        final = final.sort(dim=1)
        reverse_index = final.indices[0]

        x = x[:, duplicate_index, :, :].flatten()
        random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
        # x[random_index] = torch.randn(random_index.size())
        x[random_index] = 0
        # 1000 32 28 28
        # torch.sum(x)
        x = x.reshape(origin_shape)
        x = x[:, reverse_index, :, :]

        return x

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
        x[random_index] = 0
        x = x.reshape(origin_shape)

        return x

    def error_injection_weights(self, module, x, error_rate):
        if isinstance(module, nn.Conv2d):
            # print(module.weight.data.size())
            size = module.weight.data.size()
            total_dim = torch.zeros(size).flatten().shape[0]
            random_index = torch.randperm(total_dim)[:int(total_dim * error_rate)]
            x1 = torch.zeros(total_dim)
            x_zero = torch.zeros(size).cuda()
            x[random_index] = 1
            x1 = x1.reshape(size).cuda()
            with torch.no_grad():
                module.weight.data = torch.where(x1 == 0, module.weight.data, x_zero)
            # print(module.weight.data.size())
        x = module(x)
        return x

    def duplication(self, x_original, x_error, duplicate_index):
        x_duplicate = x_error.clone()
        x_duplicate[:, duplicate_index[:self.num_duplication], :, :] = x_original[:, duplicate_index[:self.num_duplication], :, :]
        return x_duplicate

    def forward(self, x):
        """
        :param x: input image in shape of (N, C, H, W)
        :return: out: classification score in shape of (N, Nc)
        """
        N = x.size(0)
        self.output = []

        if self.importance:
            x = nn.Parameter(x, requires_grad=True)
            x.retain_grad()  # is there any more elegant way?
            self.output.append(x)

        # x = self.error_injection_weights(self.conv1, x, self.error)
        x = self.conv1(x)
        # print(x.size())
        x = self.nonlinear(x)
        # exit()
        
        if self.importance:
            x.retain_grad()
            self.output.append(x)
        if self.entropy:
            feature = x.clone().detach()
            xtemp = torch.flatten(feature.mean(0), start_dim=1)
            entropy = -self.softmax(xtemp.permute(1, 0)) * self.logsoftmax(xtemp.permute(1, 0))
            self.layerwise_entropy_p.append(entropy.sum(0))

        '''
            Simulate error injection
        '''
        
        if self.error:
            if self.attention_mode:
                x_copy = x.clone()
                if self.error_spread:
                    #x = self.error_injection(x, self.error / 2, self.duplicate_index, is_origin=False)
                    #x_copy = self.error_injection(x_copy, self.error / 2, self.duplicate_index, is_origin=False)
                    x = self.error_injection_new(x, self.error)
                    x_copy = self.error_injection_new(x_copy, self.error)
                    #x_dup = self.duplication(x_copy, x, self.duplicate_index)
                    # x = (x + x_copy) / 2
                else:
                    # print(self.error)
                    # x_copy = self.error_injection(x_copy, self.error, self.duplicate_index, is_origin=False)
                    x = self.error_injection_new(x, self.error)
                    x_dup = self.duplication(x_copy, x, self.duplicate_index)
                x = (x + x_dup) / 2
                # x = (x + x_copy) / 2

            else:
                x = self.error_injection(x, self.error, None, is_origin=True)
        else:
            if self.attention_mode:
                x = x.permute(0, 2, 3, 1)
                x = self.fc3(x)
                x = nn.Tanh()(x)
                x = self.fc4(x)
                x = x.permute(0, 3, 1, 2)
        
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.nonlinear(x)
        if self.importance:
            x.retain_grad()
            self.output.append(x)
        x = self.pool2(x)

        flat_x = x.reshape(N, self.flat_dim)
        out = self.fc1(flat_x)
        if self.importance:
            out.retain_grad()
            self.output.append(out)
        out = self.fc2(out)
        if self.importance:
            out.retain_grad()
            self.output.append(out)
        return out



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


# SubFlow calculate importance
def cal_importance(model, train_data):
    """
    s = o^2 * H

    :param model:
    :param train_data:
    :return:
    """

    importance_list = []
    model.importance = True
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)

        # forward pass
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        for i, out in enumerate(model.output):
            hessian_approximate = copy.deepcopy(out.grad) ** 2  # grad^2 -> hessian
            importance = hessian_approximate * (out.detach().clone() ** 2)
            importance_list.append(importance.mean(dim=0))

        break
    model.importance = False
    return importance_list


def cal_entropy(model, train_data):
    """
    s = o^2 * H

    :param model:
    :param train_data:
    :return:
    """
    importance_list = []
    model.entropy = True
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)

        # forward pass
        output = model(data)
        # loss = F.cross_entropy(output, target)
        # loss.backward()

        for i, out in enumerate(model.layerwise_entropy_p):
            importance_list.append(out)

        break
    model.entropy = False
    return importance_list


# D2NN: weight sum evaluation
def weight_sum_eval(model):
    weights = model.state_dict()
    evaluation = []
    names = []
    # need to find the connection between conv and fc
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            names.append(name)

            # output input H W
            # 64 32 3 3  (4D)
            # 64 32 (2D)
            # 32 (1D)
            evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=3).sum(dim=2).sum(dim=0))
        elif isinstance(m, nn.Linear):
            names.append(name)
            # output input
            evaluation.append(weights[name + '.weight'].detach().clone().abs().sum(dim=0))
    return evaluation, names


# def error_injection(x, error_rate, duplicate_index, is_origin):
#     """
#         Simulate Error Injection.
#         :param x: tensor, input tensor (in our case CNN feature map)
#         :param error_rate: double, rate of errors
#         :return: tensor, modified tensor with error injected
#     """
#     # duplicate_index 20
#     origin_shape = x.shape
#     if not is_origin:
#         total_dim = x[:, :20, :, :].flatten().shape[0]
#     else:
#         total_dim = x[:, :32, :, :].flatten().shape[0]
#         duplicate_index = torch.arange(32).type(torch.long).to(device)
#     index = torch.arange(32).type(torch.long).to(device)
#     final = torch.stack((duplicate_index, index), dim=0)
#     final = final.sort(dim=1)
#     reverse_index = final.indices[0]
#
#     # 12
#     # 20 (smallest)
#     x = x[:,duplicate_index,:,:].flatten() # from lower to higher
#     random_index = torch.randperm(total_dim)[:int(total_dim*error_rate)]
#     x[random_index] = 0
#     x = x.reshape(origin_shape)
#     x = x[:, reverse_index, :, :]
#
#     return x


def main():
    # 1. load dataset and build dataloader
    model_dir = "./model/"
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

    if args.evaluate:
        evaluate(args.error_rate, device, test_loader)
        return

    # 2. define the model, and optimizer.
    if args.run_original:
        model = SimpleCNN().to(device)

    else:
        PATH = "./model/original/epoch-4.pt"
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(PATH), strict=False)
        model.attention_mode = True
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
            loss = nn.CrossEntropyLoss()(output, target)
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
        if args.run_original:
            torch.save(model.state_dict(), os.path.join(model_dir, 'original/epoch-{}.pt'.format(epoch)))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, 'attention/epoch-{}.pt'.format(epoch)))


def evaluate(error_rate, device, test_loader):
    print("Evaluate model with error")
    PATH = "./model/attention/epoch-4.pt"

    model = SimpleCNN(error=error_rate).to(device)
    model.load_state_dict(torch.load(PATH), strict=False)

    # print("Evaluating model without attention...")
    # # evaluate the original model without attention
    # model.evaluate_origin = True
    # test_loss, test_acc = test(model, device, test_loader)
    # print("final acc without attention: ", test_acc, "\n\n\n")
    # with open("results.txt", 'a') as f:
    #     f.write(str(args.error_rate)+"|spread|none|"+str(test_acc*100)+"\n")

    model.num_duplication = 10
    #model.error_spread = True

    print("Evaluating model with attention...")
    # Change to evaluate the model with attention
    index = torch.arange(32).type(torch.float).to(device)
    tmp = torch.sum(model.fc3.weight, dim=0) # 32 20 num_duplication
    # fc3: output * input: 20 * 32
    # sum: -> 32
    final = torch.stack((tmp, index), dim=0)
    final = final.sort(dim=1, descending=True) # smallest -> highest
    model.duplicate_index = final.indices[0]
    print(model.duplicate_index)
    model.evaluate_origin = False
    model.attention_mode = True
    test_loss, test_acc = test(model, device, test_loader)
    print("final acc with attention: ", test_acc)
    with open("results.txt", 'a') as f:
        f.write(str(args.error_rate)+"|spread|attention|"+str(test_acc*100)+"\n")

    print("Evaluating model with SubFlow importance...")
    index = torch.arange(32).type(torch.float).to(device)
    importance = cal_importance(model, test_loader)
    tmp = importance[1].sum(2).sum(1)
    final = torch.stack((tmp, index), dim=0)
    final = final.sort(dim=1, descending=True)
    model.duplicate_index = final.indices[0]
    print(model.duplicate_index)
    model.evaluate_origin = False
    model.attention_mode = True
    test_loss, test_acc = test(model, device, test_loader)
    print("final acc with attention: ", test_acc)
    with open("results.txt", 'a') as f:
        f.write(str(args.error_rate)+"|spread|importance|"+str(test_acc*100)+"\n")

    print("Evaluating model with D2NN weight-sum...")
    index = torch.arange(32).type(torch.float).to(device)
    weight_sum, _ = weight_sum_eval(model)
    tmp = weight_sum[1]
    final = torch.stack((tmp, index), dim=0)
    final = final.sort(dim=1, descending=True)
    model.duplicate_index = final.indices[0]
    print(model.duplicate_index)
    model.evaluate_origin = False
    model.attention_mode = True
    test_loss, test_acc = test(model, device, test_loader)
    print("final acc with attention: ", test_acc)
    with open("results.txt", 'a') as f:
        f.write(str(args.error_rate)+"|spread|d2nn|"+str(test_acc*100)+"\n")
        
    print("Evaluating model with entropy...")
    index = torch.arange(32).type(torch.float).to(device)
    importance = cal_entropy(model, test_loader)
    tmp = importance[0]
    final = torch.stack((tmp, index), dim=0)
    final = final.sort(dim=1, descending=False)
    model.duplicate_index = final.indices[0]
    print(model.duplicate_index)
    model.evaluate_origin = False
    model.attention_mode = True
    test_loss, test_acc = test(model, device, test_loader)
    print("final acc with attention: ", test_acc)
    with open("results.txt", 'a') as f:
        f.write(str(args.error_rate)+"|spread|entropy|"+str(test_acc*100)+"\n")



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


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # config for dataset
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--run_original', default=True, type=str2bool,
					help='train the original model')
    parser.add_argument('--evaluate', default=False, type=str2bool,
					help='Evaluate the model')
    parser.add_argument('--error_rate', type=float, default=0, metavar='M',
                        help='error_rate')
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
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device


if __name__ == '__main__':
    args, device = parse_args()
    # np.random.seed(args.seed)
    # cudnn.benchmark = False
    # cudnn.deterministic = True
    torch.manual_seed(args.seed)
    # cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    main()