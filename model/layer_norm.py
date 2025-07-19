import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "LayerNorm module"

    def __init__(self, features: int, eps: float=1e-6):
        super(LayerNorm, self).__init__()

        # Scaling param; start with identity
        self.a_2 = nn.Parameter(torch.ones(features))

        # Bias shift; start with nothing
        self.b_2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        # NORMALIZATION STEP -
        # X - mean  / stddev to normalize; plus trained params to shif the formula around
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

