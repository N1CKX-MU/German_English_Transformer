import torch


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.multihead_attention import MultiHeadAttention

def test_mha():
    mha = MultiHeadAttention(dimension=128, num_heads=8)

    x = torch.randn(4, 10, 128)
    out = mha(x, x, x)

    assert out.shape == x.shape
    print("MultiHeadAttention test passed.")

test_mha()