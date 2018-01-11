import torch as t
import torch.nn as nn

from .position_wise_nn import PositionWiseNN
from ..attention import MultiHeadAttention
from ..embedding import PositionalEmbeddings


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, pad_idx, n_layers, n_heads, h_size, k_size, v_size, dropout=0.1):
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
        super(Decoder, self).__init__()

        self.embeddings = PositionalEmbeddings('./dataloader/data/ru_embeddings.bin',
                                               vocab_size, max_len, h_size, pad_idx)

        self.layers = nn.ModuleList([
            DecoderLayer(n_heads, h_size, k_size, v_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input, condition, condition_mask=None):
        """
        :param input: An long tensor with shape of [batch_size, input_len]
        :param condition: An float tensor with shape of [batch_size, condition_len, h_size]
        :param condition_mask: An byte tensor with shape of [batch_size, condition_len]
        :return: An float tensor with shape of [batch_size, input_len, h_size]
        """

        batch_size, input_len = input.size()

        input = self.embeddings(input)

        self_mask = self.autogressive_mask(batch_size, input_len, input.is_cuda)

        if condition_mask is not None:
            condition_mask = condition_mask.unsqueeze(1).repeat(1, input_len, 1)

        out = input
        for layer in self.layers:
            out = layer(out, condition, self_mask, condition_mask)

        return out

    @staticmethod
    def autogressive_mask(batch_size, length, cuda):
        mask = t.ones(length, length).tril_(-1).byte()
        result = mask.transpose(0, 1).repeat(batch_size, 1).view(batch_size, length, length)
        if cuda:
            result = result.cuda()
        return result


class DecoderLayer(nn.Module):
    def __init__(self, n_heads, h_size, k_size, v_size, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(n_heads, h_size, k_size, v_size, dropout)
        self.out_attention = MultiHeadAttention(n_heads, h_size, k_size, v_size, dropout)
        self.position_wise = PositionWiseNN(h_size, h_size * 4, dropout)

    def forward(self, input, condition, self_mask, out_mask=None):
        """
        :param input: An float tensor with shape of [batch_size, decoder_len, h_size]
        :param condition: An float tensor with shape of [batch_size, encoder_len, h_size]
        :param self_mask: An byte tensor with shape of [batch_size, decoder_len, decoder_len]
        :param out_mask: An byte tensor with shape of [batch_size, decoder_len, encoder_len]
        :return: An float tensor with shape of [batch_size, seq_len, h_size]
        """

        result, _ = self.self_attention(q=input, k=input, v=input, mask=self_mask)
        result, _ = self.out_attention(q=result, k=condition, v=condition, mask=out_mask)
        result = self.position_wise(result)

        return result
