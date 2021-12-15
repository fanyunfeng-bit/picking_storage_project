import scipy.io as scio
import torch
import numpy as np

path1 = r'D:\研究生工作文件\研二\与石博士合作\货架分配及回归问题\code\v_2\data\Rl_10r20s_1_10.mat'
data1 = scio.loadmat(path1)
d1 = torch.Tensor(list(data1['Res_value'])).reshape(10)

path2 = r'D:\研究生工作文件\研二\与石博士合作\货架分配及回归问题\code\v_2\data\Rl_10r20s_11_20.mat'
data2 = scio.loadmat(path2)
d2 = torch.Tensor(list(data2['Res_value'])).reshape(10)

print((d1.sum() + d2.sum()) / (2 * len(d1)))
d = torch.cat((d1, d2), -1)
print(d)

net_output = torch.Tensor([1164., 1392., 1194., 1014., 1206., 1208., 1386., 1290., 1194., 996.,
                           1260., 1370., 1088., 1284., 1084., 1212., 1048., 1286., 1200., 1046.])

gap = (net_output - d) / d
print(gap)
