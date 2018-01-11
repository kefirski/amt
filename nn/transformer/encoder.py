import torch as t
import torch.nn as nn

from .position_wise_nn import PositionWiseNN
from ..attention import MultiHeadAttention
from ..embedding import PositionalEmbeddings


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, pad_idx, n_layers, n_heads, h_size, k_size, v_size, dropout=0.):
        """
        :param vocab_size: Number of words in vocabulary
        :param max_len: Max length of encoder input
        :param pad_idx: Index of padding symbol

        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(Encoder, self).__init__()

        self.pad_idx = pad_idx
        self.embeddings = PositionalEmbeddings('./dataloader/data/en_embeddings.bin',
                                               vocab_size, max_len, h_size, pad_idx)

        self.layers = nn.ModuleList([
            EncoderLayer(n_heads, h_size, k_size, v_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input):
        """
        :param input: An long tensor with shape of [batch_size, seq_len]
        :return: An float tensor with shape of [batch_size, seq_len, h_size]
        """

        mask = t.eq(input, self.pad_idx).data
        mask_unrolled = self.unroll_mask(mask, mask.size(1))

        input = self.embeddings(input)

        out = input
        for layer in self.layers:
            out = layer(out, mask_unrolled)

        return out, mask, input

    @staticmethod
    def unroll_mask(mask, size):
        return mask.unsqueeze(1).repeat(1, size, 1)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, h_size, k_size, v_size, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(n_heads, h_size, k_size, v_size, dropout)
        self.position_wise = PositionWiseNN(h_size, h_size * 4, dropout)

    def forward(self, input, mask=None):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, h_size]
        :param mask: An byte tensor with shape of [batch_size, seq_len, seq_len]
        :return: An float tensor with shape of [batch_size, seq_len, h_size]
        """

        '''
        EncoderLayer network is defined over self-attention layer, 
        thus q, k and v are all obtained as encoder input.
        '''

        result, _ = self.attention(q=input, k=input, v=input, mask=mask)
        result = self.position_wise(result)

        return result
