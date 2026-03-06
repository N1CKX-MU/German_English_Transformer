import torch 
import torch.nn as nn
from .positional_encoding import Positional_Encoding
from .feed_forward import FeedForward
from .multihead_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, dimension,num_heads,d_ff,dopout=0.1):
        super(EncoderLayer,self).__init__()

        self.attention = MultiHeadAttention(dimension,num_heads)
        self.feed_forward = FeedForward(dimension,d_ff)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.dropout1 = nn.Dropout(dopout)

    def forward(self,x,mask=None):

        attn_out = self.MultiHeadAttention(x,x,x,mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout1(ff_out))

        return x