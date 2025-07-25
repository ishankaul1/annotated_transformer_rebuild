import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    Base Encoder-Decoder arch.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        # NOTE: what are these?
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, src_mask, tgt, tgt_mask):
        """
        Take in & process masked src and target sequences
        """ 
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
