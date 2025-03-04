import torch
import numpy as np
import torch.nn as nn

class FTSVMModel(torch.nn.Module):
    def __init__(self, input_size):
        super(FTSVMModel, self).__init__()
        self.input_size = input_size
        self.dense = nn.Linear(self.input_size, 1)

    def forward(self, x):
        output = self.dense(x)
        output = torch.squeeze(output)
        return output

