from torch import nn
from model.clones import clones
from model.sublayer_connection import SublayerConnection


class DecoderLayer(nn.Module):
    """
        Self-attn, src-attn, and feed foward.
        NOTE that the major difference versus an encoder layer is the 2nd sublayer, which
        does self attention over the output of encoder stack AND the prev decoder stack.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    # NOTE: memory is encoder layer output?
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)