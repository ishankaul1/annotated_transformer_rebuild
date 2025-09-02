import math

from torch import nn


class Embeddings(nn.Module):
    """
    Use learned mebeddings to convert input token -> vector of model dimensions.
    Eg - vocab -> model dims

    Learned linear transformation + softmax converts decoder output -> next token probs.

    Embeddings layers and pre-softmax linear share same weight matrix;
    sqrt(d_model) only applied in embeddings layers.
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
