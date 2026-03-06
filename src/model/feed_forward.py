import torch 
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dimension,d_ff):
        super(FeedForward,self).__init__()

        self.linear1 = nn.Linear(dimension,d_ff)
        self.linear2 = nn.Linear(d_ff,dimension)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out