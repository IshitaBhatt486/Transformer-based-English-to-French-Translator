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