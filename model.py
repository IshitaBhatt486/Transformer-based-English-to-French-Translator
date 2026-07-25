import torch
import torch.nn as nn
import math

# Input Embeddings
class InputEmbeddings(nn.Module):
    def __init__(self, dmodel: int, vocab_size: int):
        super().__init__()

        self.dmodel = dmodel
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, dmodel)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dmodel)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, dmodel: int, seq_len: int, dropout: float):
        super().__init__()

        self.dmodel = dmodel
        self.max_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, dmodel) # matrix of size (seq_length, dmodel)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # size of tensor: (seq_length, 1)
        div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-math.log(10000.0) / dmodel))

        pe[:, 0::2] = torch.sin(position * div_term) # for even positions
        pe[:, 1::2] = torch.cos(position * div_term) # for odd positions

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)