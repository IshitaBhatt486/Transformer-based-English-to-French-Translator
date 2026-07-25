# Architecture of a Transformer
- encoder
- decoder

Both the encoder and decoder are made up of stacked blocks
Each block is made up of two key layers- an attention layer and a feed forward for MLP

- The attention layer is where all the tokens interact and find the most relevant token
- The MLP layer is where each token internally refines it's representation

For training stability, residual connections and layer normalization are present

## How it works
- a tokenizer splits the text into tokens
- the tokens are embedded(transformed into numerical representations)
- add positional information to embeddings(where each word is situated in the sentence)
- The attention layer: The tokens interact with each other
- The MLP layer: Each token is polished individually

This way we have context aware representation.

Adding Positional information is important so that the model can distinguish between `The dog bit the boy` and `The boy bit the dog`.

## What then?
- For text generation, the last token is used to predict the next token
- For sentiment analysis