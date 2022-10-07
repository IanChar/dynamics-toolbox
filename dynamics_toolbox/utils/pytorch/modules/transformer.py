import torch
from torch.nn import Module, Linear, MSELoss, ModuleList, Conv1d, Dropout, LayerNorm, parameter, GELU
import torch.nn.functional as F
import math
import numpy as np


def do_attention(query,key,value, mask= None):
    # get scaled attention scores
    attention_scores = torch.bmm(query, key.transpose(1,2))/math.sqrt(query.size(-1))
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==0, float('-inf'))
    
    attention_weights = F.softmax(attention_scores,dim=-1)
    return torch.bmm(attention_weights, value)    


def get_positional_encoding(seq_len, dim):
    
    positional_embed = torch.zeros((seq_len, dim))
    
    for t in range(seq_len):
        for i in range(dim//2):
            positional_embed[t,2*i] = np.sin(t/10000**(2*i/dim))
            positional_embed[t,2*i+1] = np.cos(t/10000**(2*i/dim))
    return positional_embed
                                             

class AttentionHead(Module):
    
    def __init__(self, embed_dim, head_dim) -> None:
        super().__init__()
        self.Wq = Linear(embed_dim, head_dim)
        self.Wk = Linear(embed_dim, head_dim)
        self.Wv = Linear(embed_dim, head_dim)
    
    def forward(self, h, mask=None):
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        outputs = do_attention(q,k,v, mask)
        return outputs
        
        
class MultiHeadAttention(Module):
    
    def __init__(self, hidden_size, num_heads, mask=None) -> None:
        super().__init__()
        self.num_heads = num_heads
        # by convention
        self.head_dim = int(hidden_size // num_heads)

        #print(hidden_size, self.head_dim, num_heads)

        self.heads = ModuleList(
            [AttentionHead(hidden_size, self.head_dim) for _ in range(num_heads)]
        )
        
        self.output = Linear(self.num_heads*self.head_dim, hidden_size)
        
    def forward(self, h, mask=None):
        x = torch.cat([head(h, mask) for head in self.heads], dim = -1)
        return self.output(x) 


    
class FeedForward(Module):
    
    # rule of thumb: hidden size of first layer 4x embddding dimension
     def __init__(self,hidden_size, inter_size, dropout_prob = 0.3) -> None:
        super().__init__()
        self.conv1 = Linear(hidden_size, 4*hidden_size)
        self.conv2 = Linear(4*hidden_size, hidden_size)
        # standard to use gelu
        self.gelu = GELU()
        self.dropout = Dropout(dropout_prob)
    
     def forward(self,x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return self.dropout(x)
   
    
    
class TransformerEncoderLayer(Module):
    
    def __init__(self, input_dim, intermediate_size, output_dim, num_heads, dropout_prob=0.3) -> None:
        super().__init__()
        hidden_size = input_dim
        output_size = output_dim
        # layer norm is prefered for transformer
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.dropout = Dropout(dropout_prob)
        self.attention = MultiHeadAttention(hidden_size, num_heads)
        self.ff = FeedForward(hidden_size ,intermediate_size, dropout_prob)
        self.out = Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        hidden = self.layer_norm1(x)
        #skip connection as in resnet
        x = x + self.dropout(self.attention(hidden))
        # skip connection
        x = x + self.ff(self.layer_norm2(x))
        # skip connection
        return self.out(x)


class TransformerDecoderLayer(Module):

    def __init__(self, hidden_size, intermediate_size, output_size, num_heads, dropout_prob=0.3 ) -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.layer_norm3 = LayerNorm(hidden_size)

        self.dropout1 = Dropout(dropout_prob)
        self.dropout2 = Dropout(dropout_prob)
        self.dropout3 = Dropout(dropout_prob)

        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.source_attention = MultiHeadAttention(hidden_size, num_heads)

        self.ff = FeedForward(hidden_size ,intermediate_size,dropout_prob)
        self.out = Linear(hidden_size, output_size)


    def forward(self, x, memory, src_mask, trg_mask):
        hidden = self.layer_norm1(x)
        x = x + self.dropout1(self.self_attention(hidden, trg_mask))
        hidden = self.layer_norm2(x)
        x = x + self.dropout2(self.source_attention(hidde, src_mask))
        hidden = self.layer_norm3(x)
        x = x + self.dropout3(hidden)
        return self.out(x)




class Embedding(Module):
    
    def __init__(self, dim, dropout_prob= 0.3) -> None:
        super().__init__()
        
        self.layer_norm = LayerNorm(dim)
        self.dropout = Dropout(dropout_prob)
        self.linear =  Linear(dim, dim)
        
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = torch.relu(x)
        return x
        
    
class TransformerEncoder(Module):
    
    def __init__(self, num_hidden, input_dim, hidden_dim, output_dim, 
                         num_heads, seq_len = 0,  dropout_prob=0.3) -> None:
        
        super().__init__()

        print(num_hidden, input_dim, hidden_dim, output_dim, 
                         num_heads)
       # self.seq_len = seq_len
        self.dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed = Embedding(self.dim,dropout_prob)

        self.layers = ModuleList(
            [TransformerEncoderLayer(input_dim, hidden_dim, input_dim
                                     ,num_heads, dropout_prob)
                                 for _ in range(num_hidden-1)]
                                 )
        self.layers.append(TransformerEncoderLayer(input_dim, hidden_dim, output_dim
                                 ,num_heads, dropout_prob))
        
    def forward(self, x):

        positional = get_positional_encoding(x.shape[1], self.dim)
        x = self.embed(x) + positional
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoder(Module):

    def __init__(self, num_hidden, hidden_size, intermediate_size, output_size, 
                         num_heads, seq_len = 0,  dropout_prob=0.3) -> None:

        super().__init__()
        self.dim = hidden_size
        self.embed = Embedding(self.dim, dropout_prob)
        self.layers = ModuleList(
            [TransformerDecoderLayer(hidden_size, intermediate_size, hidden_size 
                                     ,num_heads, dropout_prob)
                                 for _ in range(num_hidden-1)]
                                 )
        self.layers.append(TransformerDecoderLayer(hidden_size, intermediate_size, output_size
                                 ,num_heads, dropout_prob))

        def forward(self, x, memory, src_mask, trg_mask):
            positional = get_positional_encoding(x.shape[1], self.dim)
            x = self.embed(x) + positional
            for layer in self.layers:
                x = layer(x, memory,  src_mask, trg_mask)
            return x



class TransformerForPrediction(Module):
    
    def __init__(self, encoder: TransformerEncoder, dropout_prob = 0.3) -> None:
        super(TransformerForPrediction, self).__init__()
        self.encoder = encoder
        self.dropout = Dropout(dropout_prob)
        self.lin = Linear(encoder.output_dim, encoder.output_dim)
    #    self.out = Linear(encoder.output_dim, 1)

    def forward(self, x):
        x = self.encoder(x)[:,0,:]
   #     x = self.dropout(x)
   #     x = torch.mean(x, dim=1)
        x = self.lin(x)
      #  x = torch.relu(x)
        return x
      #  return self.out(x)




        