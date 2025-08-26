import torch
import math

def attention(query, key, value, mask=None, dropout=None):
    "Scaled Dot Product Attention. Just the mathematical function, not the layer"

    # Query is pretty much input
    d_k = query.size(-1)

    # Mult query by key to get transformations of each token
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Then mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    
    # Then normalize w softmax
    p_attn = scores.softmax(dim=-1)

    # Some dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
        

    # Multiply by value
    return torch.matmul(p_attn, value), p_attn