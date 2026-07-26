- Attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Positional Encodings
![Positional Encoding Formula](Images/positionalEncoding.png)
The first formula is applied in the even positions and the second formula is applied in the odd positions
Computed in log space for numerical stability
![log simplification](Images/log-simplification.png)

- Layer Normalization:
![Layer Normalization](Images/Layer-Normalization.png)

The epsilon is added in the denominator so that if the variance is 0, it prevents division by zero

alpha and beta(bias) are learnable parameters
alpha acts as a scaling factor.
bias acts as a shifting factor.

-Multi-head Attention

![Head](Images/head.png)
![Multi-head](Images/Multi-head.png)