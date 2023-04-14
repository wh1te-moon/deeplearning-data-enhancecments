import torch
import torchvision
import torchvision.transforms as transforms
# from alexnet import AlexNet
# from torchvision.models.alexnet import AlexNet
from torchvision.models.resnet import resnet34
from torch import nn, optim
from visdom import Visdom
# 定义需要的类别
classes = ('cat', 'dog')

# 数据预处理
transform = transforms.Compose(
    [transforms.Resize((128, 128)),
     transforms.AutoAugment(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
    #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 加载CIFAR-10数据集并选择特定类别
trainset = torchvision.datasets.CIFAR10(root='./cifar10/', train=True,
                                        download=True, transform=transform)
testset= torchvision.datasets.CIFAR10(root='./cifar10/', train=False,
                                        download=True, transform=transform)
# 选择只包含特定类别的数据
cat_dog_indices = []
for i in range(len(trainset)):
    if trainset[i][1] == 3:  # cifar10中cat的标签为3
        cat_dog_indices.append(i)
    elif trainset[i][1] == 5:  # cifar10中dog的标签为5
        cat_dog_indices.append(i)
trainset = torch.utils.data.Subset(trainset, cat_dog_indices)
cat_dog_indices = []
for i in range(len(testset)):
    if trainset[i][1] == 3:  # cifar10中cat的标签为3
        cat_dog_indices.append(i)
    elif trainset[i][1] == 5:  # cifar10中dog的标签为5
        cat_dog_indices.append(i)
testset = torch.utils.data.Subset(testset, cat_dog_indices)
#######################################
cat_dog_indices = []
for i in range(len(trainset)):
    if trainset[i][1] == 3:
        cat_dog_indices.append((trainset[i][0], 0))  # 将cat的标签从3改为0
    elif trainset[i][1] == 5:
        cat_dog_indices.append((trainset[i][0], 1))  # 将dog的标签从5改为1
trainset = cat_dog_indices
cat_dog_indices = []
for i in range(len(testset)):
    if testset[i][1] == 3:
        cat_dog_indices.append((testset[i][0], 0))  # 将cat的标签从3改为0
    elif testset[i][1] == 5:
        cat_dog_indices.append((testset[i][0], 1))  # 将dog的标签从5改为1
testset = cat_dog_indices
#######################################
#这个部分如果不加，则可以加载在numclasses为10的测试鲁棒性数据中
# 创建数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criteon = nn.CrossEntropyLoss().to(device)
try:
    model=torch.load("Resnet.pth")
    print("load suceess")
    pass
except:
    print("load failed")
    model=resnet34(2).to(device)
    pass
test_acc = 0
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# viz = Visdom()
if __name__=="__main__":
    for epoch in range(5000):
        for x, label in trainloader:
            x, label = x.to(device), label.to(device)
            # viz.images(x[0:16].view(-1,3,128,128), nrow=4, win='x'+str(epoch),opts=dict(jpgquality=100))
            # viz.line([loss.item()], [epoch], win='train_loss', update='append')
            # exit()
            logits = model(x)
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
                for x, label in testloader:
                    x, label = x.to(device), label.to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    total_correct += torch.eq(pred, label).float().sum().item()
                    total_num += x.size(0)
                acc = total_correct / total_num
                if acc > test_acc:
                    test_acc = acc
                    torch.save(model,"Alexnet.pth")
                print('test acc:', acc, 'epochs:', epoch)
                # exit()