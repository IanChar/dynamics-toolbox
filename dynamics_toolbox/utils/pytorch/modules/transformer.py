import torch
from torch.nn import Module, Linear, MSELoss, ModuleList, Conv1d, Dropout, LayerNorm, parameter, GELU
import torch.nn.functional as F
import math
import numpy as np
    

# not useful anymore
def do_attention(query, key, value, mask= None):
    # get scaled attention scores
    attention_scores = torch.bmm(query, key.transpose(1,2))/math.sqrt(query.size(-1))

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==0, float('-inf'))
    attention_weights = F.softmax(attention_scores, dim=-1)

    return torch.bmm(attention_weights, value)    





def get_positional_encoding(seq_len, dim):

    positional_embed = torch.zeros((seq_len, dim))
    for t in range(seq_len):
        for i in range(dim//2):
            positional_embed[t, 2*i] = np.sin(t/1e4**(2*i/dim))
            positional_embed[t, 2*i+1] = np.cos(t/1e4**(2*i/dim))
    return positional_embed
                                             

class AttentionHead(Module):
    
    def __init__(self, embed_dim, head_dim, block_size) -> None:
        super().__init__()
        self.Wq = Linear(embed_dim, head_dim)
        self.Wk = Linear(embed_dim, head_dim)
        self.Wv = Linear(embed_dim, head_dim)
        
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, block_size, block_size))

    #    joined_dim = config.observation_dim + config.action_dim + 2
  #      self.mask.squeeze()[:,joined_dim-1::joined_dim] = 0
        # add dropout and linear projection
    
    def forward(self, x):
        B, T, C = x.size()
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        attention_scores = torch.bmm(q, k.transpose(1,2))/math.sqrt(k.size(-1))
        attention_scores = attention_scores.masked_fill(self.mask[:,:T,:T]== 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)

        return torch.bmm(attention_weights, v)    
 
        
class MultiHeadAttention(Module):
    
    def __init__(self, hidden_size, num_heads, block_size) -> None:
        super().__init__()
        self.num_heads = num_heads
        embed_dim = hidden_size
        # by convention
        self.head_dim = int(embed_dim // num_heads)

        self.heads = ModuleList(
            [AttentionHead(embed_dim, self.head_dim, block_size) for _ in range(num_heads)]
        )
        
        self.output = Linear(self.num_heads*self.head_dim, embed_dim)
        
    def forward(self, h):
        x = torch.cat([head(h) for head in self.heads], dim = -1)
        return self.output(x) 


    
class FeedForward(Module):
    
    # rule of thumb: hidden size of first layer 4x embddding dimension
     def __init__(self,hidden_size, inter_size, dropout_prob = 0.3) -> None:
        super().__init__()
        self.conv1 = Linear(hidden_size, inter_size)
        self.conv2 = Linear(inter_size, hidden_size)
        # standard to use gelu
        self.gelu = GELU()
        self.dropout = Dropout(dropout_prob)
    
     def forward(self,x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        return self.dropout(x)
   
    
    
class TransformerEncoderLayer(Module):
    
    def __init__(self, input_dim, ff_dim, output_dim, num_heads, seq_len, dropout_prob=0.3) -> None:
        super().__init__()
        hidden_size = input_dim
        output_size = output_dim
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_heads, seq_len)
        self.ff = FeedForward(hidden_size, ff_dim, dropout_prob)

        self.seq_len = seq_len
        
        # not actually trying to prdict the actions, so project to output size
        # not sure what they do in the paper for this
        self.out = Linear(hidden_size, output_size)
        
    
    def forward(self, x):
        hidden = self.layer_norm1(x)
        x = x + self.attention(hidden)
        x = x + self.ff(self.layer_norm2(x))
        return self.out(x)


class TransformerDecoderLayer(Module):

    def __init__(self, hidden_size, ff_dim, output_size, num_heads, dropout_prob=0.3 ) -> None:
        super().__init__()
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.layer_norm3 = LayerNorm(hidden_size)

        self.dropout1 = Dropout(dropout_prob)
        self.dropout2 = Dropout(dropout_prob)
        self.dropout3 = Dropout(dropout_prob)

        self.self_attention = MultiHeadAttention(hidden_size, num_heads)
        self.source_attention = MultiHeadAttention(hidden_size, num_heads)

        self.ff = FeedForward(hidden_size ,ff_dim, dropout_prob)
        self.out = Linear(hidden_size, output_size)


    def forward(self, x, memory, src_mask, trg_mask):
        hidden = self.layer_norm1(x)
        x = x + self.dropout1(self.self_attention(hidden, trg_mask))
        hidden = self.layer_norm2(x)
        x = x + self.dropout2(self.source_attention(hidden, src_mask))
        hidden = self.layer_norm3(x)
        x = x + self.dropout3(hidden)
        return self.out(x)


class Embedding(Module):
    
    def __init__(self, dim, embed_dim, dropout_prob= 0.3) -> None:
        super().__init__()
        
        self.layer_norm = LayerNorm(dim)
        self.dropout = Dropout(dropout_prob)
        self.linear =  Linear(dim, embed_dim)
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = torch.relu(x)
        return x
        
    
class TransformerEncoder(Module):
    
    def __init__(self, num_hidden, input_dim, embed_dim, ff_dim, output_dim, 
                         num_heads, seq_len = 0,  dropout_prob=0.3) -> None:
        
        super().__init__()
        self.dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        #try nn embedding
        self.embed = Embedding(self.dim, embed_dim, dropout_prob)

        self.layers = ModuleList(
            [TransformerEncoderLayer(embed_dim, ff_dim, embed_dim
                                     ,num_heads, seq_len, dropout_prob)
                                 for _ in range(num_hidden)]
                                 )
       # self.layers.append(TransformerEncoderLayer(embed_dim, ff_dim, output_dim
        #                         ,num_heads, dropout_prob))
        self.seq_len = seq_len
        self.embed_dim = embed_dim
    def forward(self, x):

        positional = get_positional_encoding(x.shape[1], self.embed_dim)
        # add dropout

        x = self.embed(x) + positional
        for layer in self.layers:
            x = layer(x)
        # add layernorm
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
        self.head = Linear(encoder.embed_dim, encoder.output_dim)
        self.apply(self._init_weights)

        self.seq_len = encoder.seq_len
        self.embed_dim = encoder.embed_dim


    # taken from trajectory transformer code
    def pad_to_full_observation(self, x):
        b, t, _ = x.shape
        n_pad = (self.seq_len - t % self.seq_len) % self.seq_len
        padding = torch.zeros(b, n_pad, self.embed_dim)
        ## [ B x T' x embedding_dim ]
       # print(x.shape, padding.shape)
        x_pad = torch.cat([x, padding], dim=1)
        ## [ (B * T' / transition_dim) x transition_dim x embedding_dim ]
        x_pad = x_pad.view(-1, self.seq_len, self.embed_dim)

        return x_pad, n_pad


    def forward(self, x):
        x = self.encoder(x)[:,-1,:]
      #  b,t, _ = x.shape
      #  x = self.encoder(x)
        #print(x.shape)
     #   x, n_pad = self.pad_to_full_observation(x)
     #   x = x[:,t-1,:]
        #print(x.shape)
        x = self.head(x)
   #     x = x.reshape(b, t+n_pad, x.shape[-1])
        return x

    # no clue what why this is here
    def _init_weights(self, module):

        if isinstance(module, LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)




        