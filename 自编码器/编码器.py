import torch
import torch.nn as nn
import torch.nn.functional as func


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.line1 = nn.Linear(28 * 28 * 3, 176)
        self.line2 = nn.Linear(176, 31)
        pass

    def forward(self, input_v):
        v = func.relu(self.line1(input_v))
        v = self.line2(v)
        return v
        pass

    pass
