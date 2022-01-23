import torch
from torch import nn
import math
from typing import Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 500):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class Txt2SqlTransformer(nn.Module):
    def __init__(
        self, vocab_size=20_000, embed_dim=512, d_model=512, num_layers=4, n_head=16
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=0.05)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            activation="gelu",
            batch_first=False,
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        inp_txt,
        trg_txt,
        src_mask,
        trg_mask,
        src_padding_mask,
        trg_padding_mask,
        memory_key_padding_mask,
    ):
        inp_emb = self.positional_encoding(self.embedding(inp_txt))
        trg_emb = self.positional_encoding(self.embedding(trg_txt))
        trans_out = self.transformer(
            inp_emb,
            trg_emb,
            src_mask,
            trg_mask,
            None,
            src_padding_mask,
            trg_padding_mask,
            memory_key_padding_mask,
        )
        return self.head(trans_out)

    def encode(self, src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.embedding(src)), src_mask
        )

    def decode(self, trg, memory, trg_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.embedding(trg)), memory, trg_mask
        )

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_mask(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt == 0).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
