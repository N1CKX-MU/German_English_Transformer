import torch 
import torch.nn as nn 

class Tokenembedder(nn.Module):
    def __init__(self,vocab_size,dimension):
        super(Tokenembedder,self).__init__()

        self.embedding = nn.Embedding(vocab_size,dimension)

    def forward(self,x):
        return self.embedding(x)
    
    