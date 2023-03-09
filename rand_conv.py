import numpy as np
import torch
from torch import nn
import random
from torch.nn import functional as F


class RandConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1,seed=6389):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.seed=seed
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=int((kernel_size - 1) / 2))

    def forward(self, input):
        self.setup_seed()
        # 随机卷积提取纹理
        with torch.no_grad():
            random_convolution = self.conv
            torch.nn.init.normal_(random_convolution.weight, 0, 1 / (3 * self.kernel_size))
            # torch.nn.init.normal_(random_convolution.bias,0,1)
            output = random_convolution(input)
            # mix up
            a = torch.rand(1)
            output = a * output + (1 - a) * input
            return torch.as_tensor(output)

    def setup_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        pass
