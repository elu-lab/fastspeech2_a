#### Github [transformer.Models.py]: https://github.com/ming024/FastSpeech2/blob/master/transformer/Models.py

import numpy as np
import torch
import torch.nn as nn

import transformer.Constants as Constants
from .Layers import FFTBlock ## Not yet
from text.symbols import symbols

############################# @ transformer #################################
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)



############################# @ FastSpeech2 #################################
class Encoder(nn.Module):
    """ Encoder """

    def __init__(self, config=None):
        super(Encoder, self).__init__()

        n_position = config["max_seq_len"] + 1                  # 1000 + 1 
        n_src_vocab = len(symbols) + 1                          # 360 + 1 
        d_word_vec = config["transformer"]["encoder_hidden"]    # 256 
        n_layers = config["transformer"]["encoder_layer"]       # 4 
        n_head = config["transformer"]["encoder_head"]          # 2 
        d_k = d_v = (config["transformer"]["encoder_hidden"] // config["transformer"]["encoder_head"]) # 128  
        d_model = config["transformer"]["encoder_hidden"]       # 256 
        d_inner = config["transformer"]["conv_filter_size"]     # 1024 
        kernel_size = config["transformer"]["conv_kernel_size"] # [9, 1] 
        dropout = config["transformer"]["encoder_dropout"]      # .1 

        self.max_seq_len = config["max_seq_len"]                # 1000 
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx= Constants.PAD)
        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False,)

        self.layer_stack = nn.ModuleList([FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src_seq, mask, return_attns=False):

        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        ### Don't know here: self.training
        if not self.training and src_seq.shape[1] > self.max_seq_len:
            # print("Upper"): not printed
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            #### BASED on `else`
            # print("Bottom"): printed
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask, slf_attn_mask=slf_attn_mask )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output



############################# @ FastSpeech2 #################################
class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, config=None):
        super(Decoder, self).__init__()

        n_position = config["max_seq_len"] + 1                   # 1000 + 1 
        n_src_vocab = len(symbols) + 1                           # 360 + 1 
        d_word_vec = config["transformer"]["decoder_hidden"]     # 256 
        n_layers = config["transformer"]["decoder_layer"]        # 6
        n_head = config["transformer"]["decoder_head"]           # 2 
        d_k = d_v = (config["transformer"]["decoder_hidden"] // config["transformer"]["decoder_head"]) # 128  
        d_model = config["transformer"]["decoder_hidden"]        # 256 
        d_inner = config["transformer"]["conv_filter_size"]      # 1024 
        kernel_size = config["transformer"]["conv_kernel_size"]  # [9, 1] 
        dropout = config["transformer"]["decoder_dropout"]       # .1 

        self.max_seq_len = config["max_seq_len"]                 # 1000 
        self.d_model = d_model

        self.position_enc = nn.Parameter(get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False,)

        self.layer_stack = nn.ModuleList([FFTBlock(d_model, n_head, d_k, d_v, d_inner, kernel_size, dropout=dropout) for _ in range(n_layers)])

    def forward(self, enc_seq, mask, return_attns=False):
        ### enc_seq: output of Variance Adaptor [16, 1493, 256]
        ### mask: mel_masks: [16, 1493]

        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        # -- Forward
        ### Still Don't know here: self.training??
        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            # print("Upper")
            # -- Prepare masks
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to( enc_seq.device)
        else:
            # print("Bottom") ## printed
            # -- Prepare masks
            max_len = min(max_len, self.max_seq_len)
            # 1000
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            # slf_attn_mask: [16, 1000, 1493]
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[:, :max_len, : ].expand(batch_size, -1, -1)
            # enc_seq[:, :max_len, :] : [16, 1000, 256]
            # self.position_enc[:, :max_len, : ].expand(batch_size, -1, -1) : [16, 1000, 256]
            # dec_output : [16, 1000, 256]

            mask = mask[:, :max_len]
            # mask: [16, 1493] -> [16, 1000]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]
             # slf_attn_mask: [16, 1000, 1493] ->  [16, 1000, 1000]

        for dec_layer in self.layer_stack:
            # mask: [16, 1000]
            # slf_attn_mask: [16, 1000, 1000]
            dec_output, dec_slf_attn = dec_layer(dec_output, mask=mask, slf_attn_mask=slf_attn_mask )
            # dec_output   : [16, 1000, 256]
            # dec_slf_attn : [32, 1000, 1000]
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask
