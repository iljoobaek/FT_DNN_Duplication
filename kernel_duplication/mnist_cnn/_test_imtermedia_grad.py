import torch
import torch.nn as nn
import torch.nn.functional as F


class TestNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinear = nn.Softmax(dim=1)
        self.identity = nn.Identity()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        out = []
        # x = self.identity(x)
        x = nn.Parameter(x, requires_grad=True)
        out.append(x)
        x = self.fc1(x)
        out.append(x)
        x = self.fc2(x)
        # x = self.nonlinear(x)
        out.append(x)
        return out


def main():
    """
    y = W2 * (W1 * x + b1) + b2 (x=[1, 2])
    loss = (y - y')^2 (y'=1)
    >>> OrderedDict([('fc1.weight', tensor([[-0.5375,  0.1121], [ 0.5978,  0.6958], [-0.5703,  0.4102]])),
    >>> ('fc1.bias', tensor([-0.0475,  0.6569,  0.1748])),
    >>> ('fc2.weight', tensor([[ 0.2081, -0.5016, -0.1861]])), ('fc2.bias', tensor([0.5329]))])

    >>> intermedia variable
    >>> tensor([-0.3608,  2.6462,  0.4249], grad_fn=<AddBackward0>)
    >>> tensor([-0.9488], grad_fn=<AddBackward0>)
    >>> gradient of intermedia variable
    >>> tensor([-0.8112,  1.9552,  0.7253])
    >>> tensor([-3.8976])

    o1 = W1*x + b1
    o2 = W2*o1 + b2 = y

    d(loss)/d(o2) = d(loss)/d(y) = 2*(y-y')*d(y)/d(y) = 2*(y-y')
                = 2 * (-0.9488-1) = -3.8976

    d(loss)/d(o1) = d(loss)/d(o2) * d(o2)/d(o1) = d(loss)/d(o2) * W2
                = -3.8976 * [0.2081, -0.5016, -0.1861] = [-0.8111, 1.9550, 0.7253]

    :return:
    """
    model = TestNN()
    w = model.state_dict()
    print(w)

    output = model(torch.Tensor([1., 2.]))
    loss = F.mse_loss(output[-1], torch.Tensor([1,]))
    print("intermedia variable")
    for out in output:
        # out.requires_grad = True
        out.retain_grad()
        print(out)

    loss.backward()
    print("gradient of intermedia variable")
    for o in output:
        print(o.grad)



if __name__ == '__main__':
    main()
