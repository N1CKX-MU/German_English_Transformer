import torch 
from torch import nn




class Positional_Encoding(nn.Module):

    def __init__(self, dimension,max_len=3000):
        super(Positional_Encoding,self).__init__()

        pe = torch.zeros(max_len, dimension)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dimension, 2) * (-torch.log(torch.tensor(10000.0)) / dimension))

        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)


        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

       