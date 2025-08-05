from torch import nn
from model.layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    Residual connection followed by layernorm for a sublayer.
    Actual formula is
        LayerNorm(x + Sublayer(x))

    This impl does -
        Dropout(Sublayer(LayerNorm(x))) + x

    NOTE: Annotated transformer apply LayerNorm  on the inside rather than the outside for "simplicity"
    Wonder why it's more simple this way?
    Would the dropout still wrap sublayer in a real impl?    
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        # NOTE: sublayer not initialized here; architecture controlled by the layer itself.
        # Makes sense since it really just needs dropoout and size for init. Sublayer itself passed in for fwd pass.

        # TODO: Try pulling the LayerNorm to the outside?
        return x + self.dropout(sublayer(self.norm(x)))
    