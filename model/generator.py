from torch import nn
from torch.nn.functional import log_softmax


class Generator(nn.Module):
    """
    Standard linear & softmax gen step
    Project decoder output onto vocuabularys size,
    then find most likely word in vocab.
    """

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(in_features=d_model, out_features=vocab)

    # Assuming x needs to be a Tensor of some sort
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
