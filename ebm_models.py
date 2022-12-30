
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
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Linear(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)    

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
            #mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            #scores = scores.masked_fill(mask != 0, -1e9) ############# testing mask [09280530] wierd...

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        return output    
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dff, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model, dff, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        #x2 = self.norm_1(x) # to be commented out
        x2 = x
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = x
        #x2 = self.norm_2(x) # to be commented out
        x = x + self.dropout_2(self.ff(x2))
        return x    

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, rate):
        super().__init__()
        self.num_layers = num_layers
        self.embed = Embedder(3, d_model)
        #self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, num_heads, dff, rate), num_layers)
        self.norm = Norm(d_model)
        
    def forward(self, x, mask):
        x = self.embed(x)
        #x = self.pe(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        #return self.norm(x)
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

# [09-20]
# Minkowski Transformer
def minkowski_attention(q, k, v, mask=None):
      def minkowski_product(v1, v2):
            v2 = torch.concat([v2[..., 0:1], -v2[..., 1:]], axis=-1)
            return torch.matmul(v1, v2, transpose_b=True)

      #matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
      minkowski_qk = minkowski_product(q, k)

      # scale matmul_qk
      #dk = tf.cast(tf.shape(k)[-1], tf.float32)
      #scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

      # add the mask to the scaled tensor.
      if mask is not None:
        minkowski_qk += (torch.squeeze(mask, 1) * -1e9)  

      attention_weights = F.softmax(minkowski_qk, dim=-1)  # (..., seq_len_q, seq_len_k)

      output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
      return output

class MinkowskiEncoderLayer(nn.Module):
  def __init__(self, d_model=72, k=1e-3):
    super(MinkowskiEncoderLayer, self).__init__(name="encoder")

    self.k = k
    self.m_mlp = nn.Sequential(nn.Linear(1, d_model),
                               nn.ReLU(),
                               nn.Norm(d_model),
                               nn.Linear(d_model, d_model) ##### DEBUG ###########
                              )
    
    self.projector = nn.Sequential(nn.Linear(d_model, d_model),
                                   nn.ReLU(),
                                   nn.Norm(d_model),
                                   nn.Linear(d_model, 1)
                                  )
    
    self.e_weight_mlp = nn.Sequential(nn.Linear(d_model, d_model),
                                      nn.ReLU(),
                                      nn.Norm(d_model),
                                      nn.Linear(d_model, 1),
                                      nn.Sigmoid()
                                     )
    
    self.h_mlp = nn.Sequential(nn.Linear(2*d_model, d_model),
                               nn.ReLU(),
                               nn.Norm(d_model),
                               nn.Linear(d_model, d_model)
                              )
    
  def call(self, x, h, training, mask):

    #attn_output, _ = minkowski_attention(x, x, x, mask) # (batch_size, input_seq_len, 4)
    
    products = minkowski_product(x, x) # (batch_size, input_seq_len, input_seq_len)
    
    message = self.m_mlp(tf.expand_dims(products, axis=-1)) #similar to point-wise-feed-forward (batch_size, input_seq_len, input_seq_len, d_model)
    
    message_x = self.projector(message) # (batch_size, input_seq_len, input_seq_len, 1)
    
    message_x = torch.squeeze(message_x, -1) # (batch_size, input_seq_len, input_seq_len)
    
    if mask is not None:
         message_x += (torch.squeeze(mask, 1) * -1e9)  

    attention_weights = F.softmax(message_x, axis=-1)  # (..., input_seq_len, input_seq_len)

    attn_output = torch.matmul(attention_weights, x)  # (..., input_seq_len, 4)
    
    x = x + self.k * attn_output
    
    e_weights = self.e_weight_mlp(message) # (batch_size, seq_len, seq_len, 1)
    message_h = torch.reduce_sum(e_weights * message, -2) # (..., input_seq_len, d_model)
    
    message_h = torch.concat([h, message_h], -1)
    h = h + self.h_mlp(message_h) # a thought: we don't have input features, so we assume there are (Einstein) latent variables initilized as Gaussian Model, => LorentzVAE? (https://en.wikipedia.org/wiki/Hidden-variable_theory)

    return x, h 

#class LorentzTransformer(nn.Module):
    