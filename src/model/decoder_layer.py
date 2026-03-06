import torch 
import torch.nn as nn
from .multihead_attention import MultiHeadAttention
from .feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self,dimension,num_heads,d_ff,dropout=0.1):
        super(DecoderLayer,self).__init__()

        self.self_attention = MultiHeadAttention(dimension,num_heads)
        self.enc_dec_attention = MultiHeadAttention(dimension,num_heads)
        self.feed_forward = FeedForward(dimension,d_ff)

        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.norm3 = nn.LayerNorm(dimension)

        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_out,src_mask=None,tgt_mask=None):
        self.self_attention_out = self.self_attention(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(self.self_attention_out))   
        self.cross_attn = self.enc_dec_attention(x,enc_out,enc_out,src_mask)
        x = self.norm2(x + self.dropout(self.cross_attn))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x