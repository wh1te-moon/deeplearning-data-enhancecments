from pandas import DataFrame
import plotly
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from umap.umap_ import UMAP
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from samplepairing import samplepairing
from rand_conv import RandConv
from samplepairing import samplepairing
from rand_conv import RandConv
from mainfold_analyze import analyze
from My_python import get_name


def visualization(m1, m2, dim=3, name='temp.html'):
    plotly.offline.init_notebook_mode(connected=True)
    m1, m2 = m1.reshape(m1.size(0), -1), m2.reshape(m1.size(0), -1)
    if dim == 2:
        reducer = UMAP(n_neighbors=256, n_components=2, n_epochs=1000,
                       min_dist=0.5, local_connectivity=2, random_state=42)
        m1 = reducer.fit_transform(m1)
        data = np.concatenate((m1, m2), axis=1)
        df = DataFrame(data, columns=['x', 'y', 'label'])
        fig = px.scatter(data_frame=df, x='x', y='y', color=df['label'].astype(str))
        fig.update_traces(marker=dict(size=7))
        plotly.offline.plot(fig, filename=name + ".html")

    else:
        # m1 = PCA(n_components=None).fit_transform(m1)
        # m1=PowerTransformer(standardize=False).fit_transform(m1)
        # m1=QuantileTransformer(random_state=0,  output_distribution='normal').fit_transform(m1)
        # m1 = UMAP(n_components=20).fit_transform(m1)
        m1 = UMAP(local_connectivity=2, random_state=42, min_dist=0.1,
                  n_neighbors=1000, n_components=3, n_epochs=1000).fit_transform(m1)
        data = np.concatenate((m1, m2), axis=1)
        df = DataFrame(data, columns=['x', 'y', 'z', 'label'])
        fig = px.scatter_3d(data_frame=df, x='x', y='y', z='z', color=df['label'].astype(str))
        fig.update_traces(marker=dict(size=7))
        plotly.offline.plot(fig, filename=name + ".html")


if __name__ == '__main__':
    data_enhance = [
        [],
        # transforms.Resize((299, 299)),
        [transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip()],
        [transforms.RandomResizedCrop(32)],
        [transforms.ColorJitter()],
        [transforms.Grayscale(3)],
        [transforms.AutoAugment()]
    ]
    res = []
    result = [[], []]
    name = get_name(datasets.CIFAR10("./cifar10", True), "./imagenet.txt")
    weight = [0 for i in range(10)]
    for value in name.values():
        weight[int(value)] += 1
    for en in data_enhance:
        cifar_train = datasets.CIFAR10("cifar10", download=True, transform=transforms.Compose(en + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        # cifar_train.data = torch.as_tensor(cifar_train.data)
        # cifar_train.data = PCA(n_components=None).fit_transform(cifar_train.data.reshape(cifar_train.data.size(0), -1))
        # cifar_train = samplepairing(cifar_train)
        # cifar_train.data = np.uint8(cifar_train.data.reshape(50000, 32, 32, 3))
        x1 = DataLoader(cifar_train, cifar_train.data.shape[0])
        # for epoch in range(1):
        #         m1 = RandConv(3, 3).forward(m1)
        #         print(m1)
        #         pass
        for m1, m2 in x1:
            m1, m2 = m1.reshape(m1.size(0), -1), m2.reshape(m1.size(0), -1)
            m1 = UMAP(local_connectivity=2, random_state=42, min_dist=0.1,
                      n_neighbors=1000, n_components=3, n_epochs=1000).fit_transform(m1)
            data = np.concatenate((m1, m2), axis=1)
            df = DataFrame(data, columns=['x', 'y', 'z', 'label'])
            res.append(analyze(df, 10))
    for li in res:
        t = 0
        for i in range(10):
            t += li[i]
        result[0].append(t)
        # t = 0
        # for i in range(10):
        #     t += li[i] * weight[i]
        result[1].append(t)
