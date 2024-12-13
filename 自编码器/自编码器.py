import torch
import torch.nn as nn
import torch.nn.functional as func
from 编码器 import encoder
from 解码器 import decoder

class self_encoder(nn.Module):
    def __init__(self):
        super(self_encoder,self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()

    def forward(self,input_v):
        feature_v = self.encoder(input_v)
        decode_input_v = self.decoder(feature_v)
        return decode_input_v
