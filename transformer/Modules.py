### Github[Modules.py]: https://github.com/ming024/FastSpeech2/blob/master/transformer/Modules.py#L6

import numpy as np

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature ### 굳이 있어야 할까...? 
        # self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        # q, k, v: [bs * n_head, len_input, head_dim(=d_q = d_k = d_v)]
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature ## 일단 뜻에 따라주자

        ## mask: [bs*n_head, len_input, len_input]
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # attn = self.softmax(attn)
        attn = torch.softmax(attn, dim=2)
        output = torch.bmm(attn, v)

        return output, attn
