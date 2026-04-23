import torch
import torch.nn as nn

class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层3 → 隐藏层4
        self.fc1 = nn.Linear(3, 4)
        # 隐藏层激活
        self.sig1 = nn.Sigmoid()
        # 隐藏层4 → 输出1
        self.fc2 = nn.Linear(4, 1)
        # 输出层激活
        self.sig2 = nn.Sigmoid()
    
    # 前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = self.sig1(x)
        x = self.fc2(x)
        x = self.sig2(x)
        return x