import numpy as np
import torch
# from visdom import Visdom
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from LeNet5 import LeNet5
from alexnet import AlexNet
from torchvision.models.vgg import vgg19
from torchvision.models.inception import Inception3
from torchvision.models.resnet import resnet152
from torchvision.models.densenet import densenet161
from torch import nn, optim
from random import choice
from os import environ
from sklearn.decomposition import PCA
import re
# from vit_pytorch import ViT
from mainfold_mixup import ManifoldMixupModel

environ['CUDA_VISIBLE_DEVICES'] = choice(["1", "3"])
basic = [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
data_enhance = [
    transforms.Resize((299, 299)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    # [transforms.RandomResizedCrop(32)],
    [transforms.ColorJitter()],
    # [transforms.Grayscale(3)],
    # [transforms.AutoAugment()]
]
model_list = [
    # LeNet5,
    AlexNet,
    # vgg19,
    # Inception3,
    # resnet152,
    # densenet161,
]
# sample-pairing不能直接用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criteon = nn.CrossEntropyLoss().to(device)
for index, e in enumerate(model_list):
    test_acc = 0
    model = e(num_classes=10).to(device)
    model = ManifoldMixupModel(model, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if re.match(".*Inception3.*", model.__repr__()):
        batchsz = 64
    else:
        # batchsz = 256
        batchsz=16
    if re.match(".*Inception3.*", model.__repr__()):
        cifar_train = datasets.CIFAR10('./cifar10', True, transform=transforms.Compose(data_enhance + basic),
                                       download=True)
    else:
        cifar_train = datasets.CIFAR10('./cifar10', True, transform=transforms.Compose(basic), download=True)
    # cifar_train.data = torch.as_tensor(cifar_train.data)
    # cifar_train.data = PCA(n_components=None).fit_transform(cifar_train.data.reshape(cifar_train.data.size(0), -1))
    # cifar_train.data = np.uint8(cifar_train.data.reshape(50000, 32, 32, 3))
    cifar_test = datasets.CIFAR10('./cifar10', False, transform=transforms.Compose(data_enhance + basic), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)
    print(str(model))
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
    for epoch in range(5000):
        for x, label in cifar_train:
            x, label = x.to(device), label.to(device)
            if re.match(".*Inception3.*", model.__repr__()):
                logits = model(x).logits
            else:
                logits = model(x)
            print(logits,logits.shape)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch)
        if epoch % 3 == 0:
            model.eval()
            total_correct = 0
            total_num = 0
            with torch.no_grad():
                for x, label in cifar_test:
                    x, label = x.to(device), label.to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    total_correct += torch.eq(pred, label).float().sum().item()
                    total_num += x.size(0)
                acc = total_correct / total_num
                if acc > test_acc:
                    test_acc = acc
                    torch.save(model, str(e) + ".mdl")
                print('test acc:', acc, 'epochs:', epoch)
    print(str(e) + ".mdl" + " ok")
