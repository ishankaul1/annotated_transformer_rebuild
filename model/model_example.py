import copy

from torch import nn

from model.multi_headed_attention import MultiHeadedAttention
from model.pw_ff import PositionwiseFeedForward
from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder
from model.encoder_decoder import EncoderDecoder
from model.encoder_layer import EncoderLayer
from model.decoder_layer import DecoderLayer
from model.embeddings import Embeddings
from model.generator import Generator

c = copy.deepcopy

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Construct model from hyperparams"""
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model