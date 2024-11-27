import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        layers = []
        hidden_size = [256, 512, 256, 128]
        layers = []
        prev_size = 128
        output_size = 2
        for i, hidden_size in enumerate(hidden_size):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        val = self.net(x)
        return F.softmax(val, dim=1)
