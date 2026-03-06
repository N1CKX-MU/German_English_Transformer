import torch 
import torch.nn as nn
from .encoder_layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self,num_layers,dimension,num_heads,d_ff,dropout=0.1):
        super(Encoder,self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(dimension,num_heads,d_ff,dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self,x,mask=None):
        for layer in self.layers:
            x = layer(x,mask)

        return x