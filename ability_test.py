import numpy as np
import torch
from My_python import get_name
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import pandas


def load_imagenet_data(dataset):
    match = get_name(dataset, "./imagenet.txt")
    imagenet = datasets.ImageFolder("./imagenet/val", transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    for key, value in match.items():
        for i in range(key * 50, (key + 1) * 50):
            yield imagenet[i][0].reshape(1, 3, 32, 32), value


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("models/cifar10/AlexNet.mdl", map_location=torch.device('cpu'))
    cifar_test = datasets.CIFAR10('./cifar10', False, transform=transforms.Compose([
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    # cifar_test.data = torch.as_tensor(cifar_test.data)
    # cifar_test.data = PCA(n_components=None).fit_transform(cifar_test.data.reshape(cifar_test.data.size(0), -1))
    # cifar_test.data = np.uint8(cifar_test.data.reshape(10000, 32, 32, 3))
    cifar_test = DataLoader(cifar_test, batch_size=256, shuffle=True)
    # cifar_test = load_imagenet_data(cifar_test)
    model.eval()
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for x, label in cifar_test:
            logits = model(x)
            # print(logits)
            pred = logits.argmax(dim=1)
            # print(pred)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
        acc = total_correct / total_num
        print('test acc:', acc)
