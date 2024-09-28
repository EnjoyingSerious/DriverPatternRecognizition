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
        # print(f"经过attention后的维度是{output.shape}")
        return output


# class DecoderAttention(nn.Module):
#     def __init__(self, hidden_dimension):
#         super().__init__()
#         self.q_proj = nn.Linear(hidden_dimension, hidden_dimension)
#         self.k_proj = nn.Linear(hidden_dimension, hidden_dimension)
#         self.v_proj = nn.Linear(hidden_dimension, hidden_dimension)
#         self.hidden_dimension = hidden_dimension
        
#     def forward(self, q, k, v):
#         q = self.q_proj(q)
#         k = self.k_proj(k)
#         v = self.v_proj(v)
        
#         attn_score = torch.matmul(q, k.transpose(-2, -1))
#         scale_factor = self.hidden_dimension ** 0.5
#         attn_score = attn_score / scale_factor
        
#         attn_weights = torch.softmax(attn_score, dim=-1)
#         output = torch.matmul(attn_weights, v)
        
#         return output
        
        

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
    def __init__(self, hidden_dimension, input_dimension):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(hidden_dimension)
        self.decode_attn = EncoderAttention(hidden_dimension)
        self.ln2 = nn.LayerNorm(hidden_dimension)
        self.mlp = MLP(hidden_dimension, input_dimension)
        
    def forward(self, q):
        q = q + self.decode_attn(self.ln1(q))
        output = self.mlp(self.ln2(q))
        return output
    
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension ,num_layers = 2):
        super().__init__()
        
        self.encoders = nn.ModuleList([Encoder(input_dimension, hidden_dimension) for _ in range(num_layers)])
        self.decoders = nn.ModuleList([Decoder(hidden_dimension, input_dimension) for _ in range(num_layers)])
        
    def forward(self, encoder_input):
        
        encoder_output = None
        decoder_output = None

        for encoder, decoder in zip(self.encoders, self.decoders):
            encoder_output = encoder(encoder_input)  # 通过编码器
            decoder_output = decoder(encoder_output)  # 通过解码器
            encoder_input = decoder_output
        
        return decoder_output  # 返回最后一层解码器的输出




