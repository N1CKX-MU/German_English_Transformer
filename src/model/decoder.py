import torch 
import torch.nn as nn
from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self,num_layers,dimensions,num_heads,d_ff,dropout):
        super(Decoder,self).__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(dimensions,num_heads,d_ff,dropout)
                for _ in range(num_layers)
            ]
        )
    def forward(self,x,enc_out,src_mask=None,tgt_mask=None):
        for layer in self.layers:
            x = layer(x,enc_out,src_mask,tgt_mask)

        return x