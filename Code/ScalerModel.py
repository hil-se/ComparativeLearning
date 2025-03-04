import torch
import numpy as np
import torch.nn as nn

class ScalerModel(torch.nn.Module):
    def __init__(self):
        super(ScalerModel, self).__init__()
        self.dense1 = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        output = torch.t(x)
        output = self.dense1(output)
        output = torch.squeeze(output)
        return output
