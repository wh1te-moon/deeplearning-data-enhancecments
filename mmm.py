import torch
print(torch.cuda.get_arch_list())
a=torch.tensor([1,2])
a=a.cuda()
print(a)