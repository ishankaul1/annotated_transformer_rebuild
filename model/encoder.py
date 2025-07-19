import torch.nn as nn

from model.clones import clones
from model.layer_norm import LayerNorm

class Encoder(nn.Module):
    "Core encoder - stack of N layers"

    def __init__(self, layer: nn.Module, N: int):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        " Input and mask through each layer in order"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
