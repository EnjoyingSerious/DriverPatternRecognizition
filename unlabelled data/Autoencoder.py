import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class EncoderAttention(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.input_dimension = input_dimension
        self.c_attn = nn.Linear(input_dimension, 3*input_dimension)  # 得到qkv
        
    
    def forward(self, x):
        qkv = self.c_attn(x)
        q, k, v = torch.split(qkv, self.input_dimension, dim=2)
        
        # 获得分数并将其缩放
        attn_score = torch.matmul(q, k.transpose(-2, -1))
        scale_factor = self.input_dimension ** 0.5
        attn_score = attn_score / scale_factor
        
        # 计算softmax权重
        attn_weights = torch.softmax(attn_score, dim=-1)
        
        output = torch.matmul(attn_weights, v)
        print(f"经过attention后的维度是{output.shape}")
        return output


class MLP(nn.Module):
    def __init__(self, input_dimension, out_dimension):
        super().__init__()
        self.c_fc = nn.Linear(input_dimension, 4*input_dimension)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*input_dimension, out_dimension)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_proj(self.gelu(x))
        return x


class Encoder(nn.Module):   
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_dimension)
        self.encode_attn = EncoderAttention(input_dimension)
        self.ln2 = nn.LayerNorm(input_dimension)
        self.mlp = MLP(input_dimension, output_dimension)
    
    def forward(self, x):
        x = x + self.encode_attn(self.ln1(x))
        x = self.mlp(self.ln2(x))
        return x
        
        
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()