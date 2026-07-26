## Layer Normalization
Each item's mean and variance is independently calculated. Then we calculate their new values based on these.

Normalization is a DL technique that is used to stabilize training by normalizing inputs across feature dimensions for each individual sample

keepdim=True ensures the output shape remains compatible for broadcasting.
dim=-1, usually the feature/embedding dimension

$(x - \mu) / (\sigma + \epsilon)$: Standardizes features so they have a mean of 0 and variance of 1.self.alpha * ... + self.bias: Scales and shifts the normalized features, allowing the network to learn the optimal variance and mean dynamically during training.

Unlike Batch Normalization, which normalizes across the batch dimension, Layer Normalization normalizes across the feature dimension independently for each sample. This makes it independent of batch size and ideal for sequential tasks like Natural Language Processing (NLP).
