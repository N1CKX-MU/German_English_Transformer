import torch
import torch.nn as nn
import math 

class MultiHeadAttention(nn.Module):

    def __init__(self, dimension, num_heads):
        super(MultiHeadAttention,self).__init__()

        assert dimension % num_heads == 0, "Dimension must be divisible by number of heads"

        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads

        self.query_linear = nn.Linear(dimension, dimension)
        self.key_linear = nn.Linear(dimension, dimension)
        self.value_linear = nn.Linear(dimension, dimension)
        self.out_linear = nn.Linear(dimension, dimension)

    def forward(self, query,key,value,mask=None):
        batch_size = query.size(0)

        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)

        K = K.view(batch_size, -1, self.num_heads, self.head_dim)

        V = V.view(batch_size, -1, self.num_heads, self.head_dim)

        # Transpose to get shape 
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        energy = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)


        if mask is not None:
            energy = energy.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, V)

        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.dimension)

        out = self.out_linear(out)

        return out

