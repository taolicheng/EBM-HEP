
import copy
import math
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.optim as optim

import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)
        
class MLPJet(nn.Module):
    def __init__(self, input_dim=80, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
                spectral_norm(nn.Linear(input_dim, 512)),
                nn.ReLU(),
                spectral_norm(nn.Linear(512, 512)),
                nn.ReLU(),
                spectral_norm(nn.Linear(512, 128)),
                nn.ReLU(),
                spectral_norm(nn.Linear(128, 64)),
                nn.ReLU(),
                spectral_norm(nn.Linear(64, 1))
        )
        
    def forward(self, x):
        x = self.mlp(x)
        return x

class Embedder(nn.Module):
    def __init__(self, d_in, d_model):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        
    def forward(self, x):
        return self.embed(x)    

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
            #mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.h = num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k)
        
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        v_out = attention(q, k, v, self.d_k, mask, self.dropout)
        
        v_out = v_out.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(v_out)
    
        return output    
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout = 0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.act(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        
        self.attn = MultiHeadAttention(num_heads, d_model)
        self.ff = FeedForward(d_model, dff, dropout)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x0 = x
        #x = self.norm_1(x)
        x = self.attn(x,x,x,mask)
        x = x0 + self.dropout_1(x)
        
        x0 = x
        #x = self.norm_2(x)
        x = self.ff(x)
        x = x0 + self.dropout_2(x)
        return x    
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = Embedder(3, d_model)
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask):
        x = self.embedding(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        #x = self.norm(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, num_layers=3, d_model=128, num_heads=8, dff=256, rate=0.1, n_output=1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        self.mlp = nn.Sequential(
                nn.Linear(d_model, 500),
                Swish(),
                nn.Linear(500, 500),
                Swish(),
                nn.Linear(500, n_output)
        )
        
    
    def _create_padding_mask(self, seq):
        seq = torch.sum(seq, 2)
        seq = torch.eq(seq, 0)
        #seq = tf.cast(torch.eq(seq, 0), tf.float32)
        seq = torch.unsqueeze(seq, 1)
        seq = torch.unsqueeze(seq, 1)
        
        return seq  # (batch_size, 1, 1, seq_len)  
    
    def forward(self, x, mask=None):
        x = x.view(x.shape[0], -1, 3)
        if mask is None:
            mask = self._create_padding_mask(x) 
            
        e_outputs = self.encoder(x, mask)
        e_outputs = torch.sum(e_outputs, 1)
        
        output = self.mlp(e_outputs)
        return output

'''
def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
            #mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output   
'''
    