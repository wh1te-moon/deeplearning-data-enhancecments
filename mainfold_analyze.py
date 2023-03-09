import random

from pandas import DataFrame
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from umap.umap_ import UMAP
import numpy as np
from samplepairing import samplepairing
from rand_conv import RandConv
from samplepairing import samplepairing
from math import log
from statistics import mean
from My_python import get_name


def normalization(df):
    need = [[df.min()["x"], df.max()["x"]], [df.min()["y"], df.max()["y"]], [df.min()["z"], df.max()["z"]]]
    new = [[], [], []]
    for x, y, z in df[["x", "y", "z"]]:
        need[0].append((x - need[0][0]) / (need[0][1] - need[0][0]))
        need[1].append((y - need[1][0]) / (need[1][1] - need[1][0]))
        need[2].append((z - need[2][0]) / (need[2][1] - need[2][0]))
    df["x"] = new[0]
    df["y"] = new[1]
    df["z"] = new[2]


def analyze(df, num_classes=10):
    res = []
    for i in range(num_classes):
        temp_df = df[df.label == i]
        temp_df.index = [i for i in range(len(temp_df))]
        temp_res = [0, 0, 0]
        for index, dim in enumerate(["x", "y", "z"]):
            temp_df.sort_values(by=dim)
            temp = []
            for j in range(len(temp_df) - 1):
                temp.append(temp_df[dim][j + 1] - temp_df[dim][j])
            m = mean(temp)
            for r in temp:
                temp_res[index] += (r - m) ** 2
            temp_res[index] /= len(temp_df)
        res.append(mean(temp_res))
    return res


def all_analyze(df):
    pass


if __name__ == "__main__":
    pass
    # df = DataFrame()
    # df["x"] = [i for i in range(1000)]
    # df["y"] = [i for i in range(1000)]
    # df["z"] = [i for i in range(1000)]
    # df["label"] = [i for i in range(10) for j in range(100)]
    # re=analyze(df,10)
    # res = [[3.069388669523475,
    #         3.509727959614645,
    #         3.558882634892844,
    #         3.2463516735649804,
    #         2.870772007595088,
    #         2.7127256721970636,
    #         3.0275949635505066,
    #         2.904116641726496,
    #         2.6553572042052767,
    #         2.4661815588579965],
    #        [3.136804717135074,
    #         3.1206049390650668,
    #         3.3077599469830417,
    #         3.322613144417908,
    #         2.8030366345661357,
    #         3.06107396306555,
    #         2.585227710639067,
    #         2.911816281536913,
    #         2.8975344402635415,
    #         3.039967679308186],
    #        [3.069388669523475,
    #         3.509727959614645,
    #         3.558882634892844,
    #         3.2463516735649804,
    #         2.870772007595088,
    #         2.7127256721970636,
    #         3.0275949635505066,
    #         2.904116641726496,
    #         2.6553572042052767,
    #         2.4661815588579965],
    #        [3.722234057833107,
    #         3.6191632836211785,
    #         3.7005661186914796,
    #         3.5048897999351665,
    #         3.0396816119194847,
    #         3.015561850382343,
    #         3.312815443102113,
    #         2.9923855039503118,
    #         3.1920799242683295,
    #         2.589778490551192],
    #        [7.115237376707277,
    #         6.099618213385788,
    #         5.509526939159244,
    #         5.353143648463377,
    #         5.055044915625614,
    #         4.812383101115946,
    #         5.247829787787017,
    #         5.4973051695973085,
    #         6.608302815954578,
    #         6.544744249935868]]
    # x1 = datasets.CIFAR10("./cifar10", True)
    # name = get_name(x1, "./imagenet.txt")
    # weight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # for value in name.values():
    #     weight[int(value)] += 1
    # result = []
    # for li in res:
    #     t = 0
    #     for i in range(10):
    #         t += li[i] * weight[i]
    #     result.append(t)
