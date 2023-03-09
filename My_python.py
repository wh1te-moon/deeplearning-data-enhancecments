import os
import re
import time
from torchvision import datasets


def time_count(is_return=False):
    def time_count1(func):
        def time_count2(*args, **kwargs):
            start_time = time.time()
            if is_return:
                res = func(*args, **kwargs)
                end_time = time.time()
                print('/运行时长:', end_time - start_time, '/')
                return res
            else:
                func(*args, **kwargs)
                end_time = time.time()
                print('/运行时长:', end_time - start_time, '/')
            pass
        return time_count2
    return time_count1
    pass


@time_count(True)
def get_path(path1, pattern):
    """
    获取目录下所有符合正则表达式(目录名也可以)的文件路径(超大的目录也可以),并返回一个结果列表
    :param path1: 目标路径
    :param pattern: 条件正则表达式
    :return:list
    """
    list1 = []  # 存放迭代器
    list2 = []  # 存放符合pattern的结果
    try:
        list1.append(os.scandir(path1))
    except WindowsError:
        print('无访问权限')
    i1 = 0  # while的辅助计数参数
    while i1 < len(list1):
        for i2 in list1[i1]:
            # print(i2.path)
            print(i1, len(list1))
            if os.path.isdir(i2.path):
                try:
                    list1.append(os.scandir(i2.path))
                except WindowsError:
                    print('无访问权限')
            elif os.path.isfile:
                if re.match(pattern, i2.path):
                    list2.append(i2.path)
                elif not re.match(pattern, i2.path):
                    pass
                pass
            pass
        # list1[i1] = None  # 可以省内存,但会多消耗1/100?的时间
        i1 += 1
    return list2


def get_name(datasets, root):
    """
    给定一个datasets,在文件中寻找这个datasets class 的字符串的匹配,返回id与imagenet的索引的列表的字典
    """
    match = {}
    with open(root, "r+", encoding="utf-8") as file:
        s = file.readlines()
        for key, value in datasets.class_to_idx.items():
            pattern = ".*[^a-zA-Z]" + key + "[^a-zA-Z].*"
            for i in s:
                r = re.match(pattern, i)
                if r:
                    r = r.group().replace("\n", "").replace("\t", "").strip()
                    r = int(re.search("[0-9]+", r).group())
                    match[r] = value
        return match


if __name__ == '__main__':
    pass
    # result = get_path(r'E:\m_cloud\projects\Deeplearning_Data_Enhanced\deeplearning-data-enhancecments\imagenet\val', r".*\(1\).*")
    # print(len(result))
    # for item in result:
    #     print(item)
    #     os.remove(item)
    x1 = datasets.CIFAR10("./cifar10", True)
    print(get_name(x1, "./imagenet.txt"))
