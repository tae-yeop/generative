import torch.nn as nn


t = nn.ModuleList([])
t.append(nn.Conv2d(1,3,3,1,1))

print(t)