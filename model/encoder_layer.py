from torch import nn

from model.clones import clones
from model.sublayer_connection import SublayerConnection

class EncoderLayer(nn.Module):
    """
    Single Encoder layer.
    Made up of self-attn and feed forward.
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 1 sublayer for self-attn, 1 for feed fwd. 
        # Basically just create 2 of the same size & same dropout setting.
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # Interesting; wrap a lambda since x needs to be expanded to 3 inputs for self attn?
        # Why?
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


