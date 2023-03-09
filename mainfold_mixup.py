import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from LeNet5 import LeNet5


def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot


bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)


class ManifoldMixupModel(nn.Module):
    def __init__(self, model, num_classes=10, alpha=1):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.lam = None
        self.num_classes = num_classes
        # 选择需要操作的层，在ResNet中各block的层名为layer1,layer2...所以可以写成如下。其他网络请自行修改
        self.module_list = []
        for n, m in self.model.named_modules():
            # if 'conv' in n:
            if n[:-1] == 'layer':
                self.module_list.append(m)

    def forward(self, x, target=None):
        if target is None:
            out = self.model(x)
            return out
        else:
            if self.alpha <= 0:
                self.lam = 1
            else:
                self.lam = np.random.beta(self.alpha, self.alpha)
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            target_onehot = to_one_hot(target, self.num_classes)
            target_shuffled_onehot = target_onehot[self.indices]
            if k == -1:
                x = x * self.lam + x[self.indices] * (1 - self.lam)
                out = self.model(x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
            target_reweighted = target_onehot * self.lam + target_shuffled_onehot * (1 - self.lam)

            loss = bce_loss(softmax(out), target_reweighted)
            return out, loss

    def hook_modify(self, module, input, output):
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        return output


"""
net = ResNet18()
net = ManifoldMixupModel(net,num_classes=10, alpha=args.alpha)
def train(epoch):
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, loss = net(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(epoch):
    net.eval()
    with torch.no_grad():
	    for batch_idx, (inputs, targets) in enumerate(testloader):
	        inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)


"""
if __name__ == "__main__":
    model = LeNet5(10)
    model = ManifoldMixupModel(model, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    cifar_train = datasets.CIFAR10('./cifar10', True, transform=transforms.ToTensor(), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=256, shuffle=True)
    for epoch in range(100):
        for x, label in cifar_train:
            logits, loss = model(x, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch)