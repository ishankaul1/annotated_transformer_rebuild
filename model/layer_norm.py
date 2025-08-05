import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm module. Reduces training time substantially by normalizing activations.

    The scaling (a_2) and bias (b_2) params are learned transformations that work 
    well given the normalized input distribution.

    Key benefits:
    - Reduces vanishing/exploding gradients by maintaining reasonable activation scales
    - Creates smoother loss surface for more stable optimization
    - Makes training less sensitive to initialization and learning rates
    
    Intuition: Acts as a 'reset button' at each layer, preventing the accumulation 
    of problematic scaling effects that compound through network depth.

    Read more: https://arxiv.org/abs/1607.06450
    """


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
        # X - mean  / stddev to normalize; plus trained bias param for shift
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

