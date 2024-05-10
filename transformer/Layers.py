from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward

### 할 게 없어지네..?

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):

        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        ## enc_output: [16, 90, 256]

        ## mask when Encoder: src_mask
        ## mask Shape: [16, 90,]
        ## mask.unsqueeze(-1) Shape:[16, 90, 1]
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)
        
        enc_output = self.pos_ffn(enc_output)

        ## mask when Encoder: src_mask
        ## mask Shape: [16, 90,]
        ## mask.unsqueeze(-1) Shape:[16, 90, 1]
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn

############################# @ fastspeech2 ###################################
### Just Conv1D?
class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear", ### NoWhere? 
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal
    

############################# @ fastspeech2-PostNet ###################################
class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels=80,
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
    ):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,                             ## 80
                    postnet_embedding_dim,                      ## 512
                    kernel_size=postnet_kernel_size,            ## 5
                    stride=1,                               
                    padding=int((postnet_kernel_size - 1) / 2), ## 2
                    dilation=1,                 
                    w_init_gain="tanh",                          ### NoWhere? 
                ),
                nn.BatchNorm1d(postnet_embedding_dim),       
            )
        )

        ## 4 Layers
        for i in range(1, postnet_n_convolutions - 1): ## range(1, 4)
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,                     ## 512
                        postnet_embedding_dim,                     ## 512
                        kernel_size=postnet_kernel_size,           ## 5
                        stride=1,   
                        padding=int((postnet_kernel_size - 1) / 2), ## 2
                        dilation=1,
                        w_init_gain="tanh",                         ### NoWhere? 
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,                     ## 512
                    n_mel_channels,                            ## 80
                    kernel_size=postnet_kernel_size,           ## 5
                    stride=1,     
                    padding=int((postnet_kernel_size - 1) / 2), ## 2
                    dilation=1,
                    w_init_gain="linear",                       ### NoWhere? 
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        ## x(=mel_output) from mel_linear after Decoder
        ## x(=mel_output): [16, 1000, 80]

        x = x.contiguous().transpose(1, 2)
        ## x(=mel_output): [16, 80, 1000]
        
        ### what is or Where is self.training?
        for i in range(len(self.convolutions) - 1):
            ## i : 0 ~ 4
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)

        ## Last Layer is 'linear'
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        ## x(=mel_output): [16, 80, 1000] -> [16, 10000, 80]
        return x
