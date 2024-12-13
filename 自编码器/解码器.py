import torch
import torch.nn as nn
import torch.nn.functional as func


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.line1 = nn.Linear(31, 176)
        self.line2 = nn.Linear(176, 28*28*3)
        pass

    def forward(self, feature_v):
        v = func.relu(self.line1(feature_v))
        v = torch.sigmoid(self.line2(v))
        return v
        pass

    pass








