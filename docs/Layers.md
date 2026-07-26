# Layers

## Input Embedding Layer
Input: Sentence as tokens
Output: Embedding(vector of size 512)
It converts discrete word/token IDs (integers) into dense, continuous vector representations and scales them.(a mapping between a number and a vector.) `nn.embedding` has been used to create this layer.

- dmodel: The size of the vector representation for each token (dimensions)
- vocab_size: The total number of unique tokens in the vocabulary

- `nn.Embedding(vocab_size, dmodel)` creates a lookup table of shape `(vocab_size, dmodel)`. Each token ID maps to a trainable vector of size dmodel

- Foward Pass function: `self.embedding(x)`: Takes an input tensor x containing token IDs and retrieves their corresponding `dmodel` dimensional vectors.
It is multiplied with sqrt of dmodel to scale it. This is done so that the variance of the embeddings is closer to 1.
The value of positional encodings tends to have a higher variance (since they are calculated using cosine and sine functions which have a range of [-1,1] and their variance is around 0.5 on avg). Thus, the positional values will be to strong relative to the semantic values and will 'dilute' it. This is why we scale the semantic value by multiplying it with the sqrt of dmodel.
(The square root is taken because:
if a random variable X is scaled by a constant c, the original variance is multiplied by c^2.
Not taking the square root can cause the variance to go up by a very large value that can cause exploding gradients during backpropagation)

## Positional Encodings

Unlike RNNs or CNNs, Transformers process all tokens in a sequence simultaneously in parallel. Because self-attention has no inherent sense of order (it operates on sets of vectors), swapping word positions would yield the exact same self-attention output.

Final Embedding = Token Embedding + Positional Encoding

We create a positional vector and add it to the embedding that was created in the previous step

PositionalEncoding inherits all functionalities of PyTorch’s base neural network class nn.Module

- seq_length: max length of the sentence- A column vector representing token indices from 0 to seq_len - 1 with shape (seq_len, 1)
- div_term: the divisor
- dropout: to reduce overfitting

We create a matrix of size (seq_length, dmodel) to hold the positional encodings

- position: holds the position of a word inside the sentence(range: 0, n-1)
- div_term: deonominator of the formula
- register_buffer: Saves pe as part of the module's state so it moves automatically to GPU/CPU with the model, but tells PyTorch that this matrix contains non-learnable fixed parameters (no gradients needed).
- self.dropout(x): Applies dropout to the combined embeddings for regularization.

## Feed Forward Layer

We apply linear->relu->dropout->linear

## Multi-head Attention block

h= number of heads
d_k= depth
wo= output matrix

mask: if we dont want a word to interact with other words(hide the attention of those words). The mask can be used to replace filler words(padding used to reach the required sequeence length)

we split these along the embedding dimension not the sequence dimension so that each head has access to the full sentence but a different part of the embedding of each word

## Residual Connection
- the connection that allows us to skip

## Encoder
- src_mask: mask applied to input- reason: hide the interaction of padding  words with other words
- we have n encoders

## Decoder
- the output embeddings of the transformer are the same as the input embeddings(so the same class is initialized twice)
- we have three residual connections
- Cross Attention: crossing two kinds of different objects together and matching them
self and cross attention blocks are the same but have different parameters
x=output of decoder
src_mask: source language- output of the encoder
target_mask: target language- output of the decoder

## Linear Layer

## Transformer
- we have source embedding for the source language
- we have target embedding for the target language
we have multiple forward methods instead of just one because during inference, we can reuse the output as the input- we do not need to recalculate it every time
also keeping it seperate helps during visualization

