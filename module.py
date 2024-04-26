import pdb
import math
import copy
import os
import sys
import psutil
from tqdm import tqdm
import argparse
import logging 
from logging import info as DEBUG
from logging import warning as INFO
from logging import error as ERROR
logging.basicConfig(level=30,format='[log: %(filename)s line:%(lineno)d] %(message)s')
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.num_attention_heads = args.num_heads
        self.attention_head_size = int(args.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(args.hidden_size, self.all_head_size)
        self.key = Linear(args.hidden_size, self.all_head_size)
        self.value = Linear(args.hidden_size, self.all_head_size)

        self.out = Linear(args.hidden_size, args.hidden_size)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        # DEBUG(f'x.shape: {x.shape}')
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # DEBUG(f'new_x_shape: {new_x_shape}')
        x = x.view(*new_x_shape)
        # DEBUG(f'x.shape: {x.shape}')
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        DEBUG(f'hidden_states.shape: {hidden_states.shape}')
        mixed_query_layer = self.query(hidden_states)
        DEBUG(f'mixed_query_layer.shape: {mixed_query_layer.shape}')
        mixed_key_layer = self.key(hidden_states)
        DEBUG(f'mixed_key_layer.shape: {mixed_key_layer.shape}')
        mixed_value_layer = self.value(hidden_states)
        DEBUG(f'mixed_value_layer.shape: {mixed_value_layer.shape}')

        query_layer = self.transpose_for_scores(mixed_query_layer)
        DEBUG(f'query_layer.shape: {query_layer.shape}')
        key_layer = self.transpose_for_scores(mixed_key_layer)
        DEBUG(f'key_layer.shape: {key_layer.shape}')
        value_layer = self.transpose_for_scores(mixed_value_layer)
        DEBUG(f'value_layer.shape: {value_layer.shape}')

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        DEBUG(f'attention_scores.shape: {attention_scores.shape}')
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        DEBUG(f'attention_scores.shape: {attention_scores.shape}')
        attention_probs = self.softmax(attention_scores)
        DEBUG(f'attention_probs.shape: {attention_probs.shape}')
        weights = attention_probs 
        DEBUG(f'weights.shape: {weights.shape}')
     
        context_layer = torch.matmul(attention_probs, value_layer)
        DEBUG(f'context_layer.shape: {context_layer.shape}')
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        DEBUG(f'context_layer.shape: {context_layer.shape}')
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        DEBUG(f'{context_layer.size()[:-2]} {(self.all_head_size,)} {self.all_head_size}')
        DEBUG(f'new_context_layer_shape: {new_context_layer_shape}')
        context_layer = context_layer.view(*new_context_layer_shape)
        DEBUG(f'context_layer.shape: {context_layer.shape}')
        attention_output = self.out(context_layer)
        DEBUG(f'attention_output.shape: {attention_output.shape}')
     
        # pdb.set_trace()
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, args):
        super(Mlp, self).__init__()
        self.fc1 = Linear(args.hidden_size, args.mlp_dim)
        self.fc2 = Linear(args.mlp_dim, args.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(args.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, args):
        super(Block, self).__init__()
        self.hidden_size = args.hidden_size
        self.attention_norm = LayerNorm(args.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(args.hidden_size, eps=1e-6)
        self.ffn = Mlp(args)
        self.attn = Attention(args)

    def forward(self, x):
        DEBUG(f'x: {x.shape}')
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        DEBUG(f'x: {x.shape}')
        DEBUG(f'weights: {weights.shape}')
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    
if __name__ == '__main__':
    logging.getLogger().setLevel(20)

    x = torch.randn(10, 197, 768)

    # att = Attention(args, vis=True)
    # y, weights = att(x)
    # DEBUG(f'y.shape: {y.shape}')
    # DEBUG(f'weights.shape: {weights.shape}')

    block = Block(args)
    y, weights = block(x)
    DEBUG(f'y.shape: {y.shape}')
    DEBUG(f'weights.shape: {weights.shape}')