from dataclasses import dataclass

import torch


@dataclass
class Config:

    max_seq_len = 2048
    d_model = 512
    hidden_dim = 2048
    n_layers = 6
    n_heads = 8
    d_qk = d_v = 64

    epochs = 300
    batch_size = 32

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    UNK_SYM, PAD_SYM, BOS_SYM, EOS_SYM = '<unk>', '<pad>', '<bos>', '<eos>'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
