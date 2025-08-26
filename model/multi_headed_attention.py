from torch import nn
from model.clones import clones
from model.attention import attention

class MultiHeadedAttention(nn.Module):
    """MultiHeadedAttention impl.
    Take in the model size & number of heads; split dims in to separate heads, and run attention over each.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        # d_v = d_k = d_q
        self.d_k = d_model // h
        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4) # Why hardcode 4?
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # NOTE: q, k, v are not actualy q, k, v yet. But they will be transformed into.

        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches=query.size(0)

        # All linear projections in batch from d_model -> h * d_k
        # Eg - apply assigned portions of the linear layer to assigned portions of q, k, v to get their actual transformed q, k, v.
        # Note that this automatically partitions the 'space' in to 'h' groups using clever 'view' tricks!
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

         # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value
        return self.linears[-1](x)





