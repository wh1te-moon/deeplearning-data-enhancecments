"""
import torch
from vit_pytorch import ViT
# 创建ViT模型实例
v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)
# 随机化一个图像输入
img = torch.randn(1, 3, 256, 256)
# 获取输出
preds = v(img) # (1, 1000)
"""
