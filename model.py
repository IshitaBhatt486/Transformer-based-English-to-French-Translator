import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, dmodel: int, vocab_size: int):
        super().__init__()

        self.dmodel = dmodel
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, dmodel)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.dmodel)

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

        pe = pe.unsqueeze(0) # (1, seq_length, dmodel)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, dmodel)
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardLayer(nn.Module):
    def __init__(self, dmodel: int, dff: int, dropout: float) -> None:
        super().__init__()

        self.dmodel = dmodel
        self.dff = dff
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(dmodel, dff) #w1 and b1
        self.linear2 = nn.Linear(dff, dmodel) #w2 and b2

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, dmodel: int, h: int, dropout: float) -> None:
        super().__init__()

        self.dmodel = dmodel
        self.h = h

        self.dropout = nn.Dropout(dropout)

        assert dmodel % h == 0, "dmodel is not divisible by the number of heads"

        self.d_k = dmodel // h

        self.wq = nn.Linear(dmodel, dmodel)
        self.wk = nn.Linear(dmodel, dmodel)
        self.wv = nn.Linear(dmodel, dmodel)

        self.wo = nn.Linear(dmodel, dmodel)

    @staticmethod
    def attention(query, key, value, mask, dropout):

        dk = query.size(-1)

        # (Batch size, h, seq_len, d_k) -> (Batch size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        #Batch size, sequence length, dmodel
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        #batch size, sequence length, dmodel -> batch size, sequence length, h, d_k -> batch size, h, sequence length, d_k
        query= query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        #(Batch size, h, sequence length, d_k) -> (Batch size, sequence length, h, d_k) -> (Batch size, sequence length, dmodel)
        x=x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.d_k)

        # (Batch size, sequence length, dmodel) -> (Batch size, sequence length, dmodel)
        return self.wo(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList, num_layers: int):
        super().__init__()

        self.layers = layers
        self.num_layers = num_layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections=nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, dmodel: int, vocab_size: int) -> None:
        super().__init__()

        self.Projection_layer = nn.Linear(dmodel, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, dmodel) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.Projection_layer(x), dim=-1)

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src,src_mask):

        src= self.src_embedding(src)
        src= self.src_pos(src)

        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt= self.tgt_embedding(tgt)
        tgt= self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

def build_Transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,dmodel: int = 512, N: int =6, h: int =8, dropout: float=0.1, dff: int = 2048) -> Transformer:

    src_embedding = InputEmbeddings(dmodel, src_vocab_size)
    tgt_embedding = InputEmbeddings(dmodel, tgt_vocab_size)

    src_pos = PositionalEncoding(dmodel, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(dmodel, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(dmodel, h, dropout)
        feed_forward_block = FeedForwardLayer(dmodel, dff, dropout)
        encoder_block=EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(dmodel, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(dmodel, h, dropout)
        feed_forward_block = FeedForwardLayer(dmodel, dff, dropout)
        decoder_block=DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks), N)
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(dmodel, tgt_vocab_size)

    transformer= Transformer(encoder, decoder,src_embedding,tgt_embedding,src_pos,tgt_pos,projection_layer)

    # initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
