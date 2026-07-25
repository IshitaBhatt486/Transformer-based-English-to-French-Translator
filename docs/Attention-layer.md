# The Attention Layer

The attention layer creates three different representations of each token:
- Query: What are we looking for?
- Key: What we know
- Value: Actual value

To decide which tokens are relevant, we take the dot product of the query of a token and the key of all other tokens

Then, normalization is done (ex: with a softmax function) to turn these dot product results into attention weights. The higher value is giver more "attention" and the lower value is given less "attention"

Finally, an updated representation of each token is created which is equal to the weighted sum of all the values vectors, where the weights are the attention scores.

This way, we get a context-aware representation of the token

## Mathematics
The paper combines the vectors into a matrix and performs the dot product and weighted sums simultaneously(the softmax function)
Using a matrix helps with parallel calculations and speeds up the process
Every token communicates with every other token in one single set of parallel matrix operations. This way it manages to be efficient.
