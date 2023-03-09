from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class p(nn.Module):
    def __init__(self):
        super(p, self).__init__()

    def forward(self, x):
        print(x.shape)
