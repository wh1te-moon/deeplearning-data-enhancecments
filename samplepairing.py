from random import choice


def samplepairing(datasets):
    """
    先要执行好to_tensor操作
    然后扩增
    再使用时要ToPltImage
    """
    data = datasets.data
    label = datasets.targets
    n_data = []
    n_label = []
    for x1 in range(datasets.data.shape[0]):
        for x2 in range(datasets.data.shape[0]):
            if x1 != x2:
                n1 = (data[x1] + data[x2]) / 2
                n2 = choice((label[x1], label[x2]))
                n_data.append(n1)
                n_label.append(n2)
    datasets.data = data
    datasets.targets = label
    return datasets
