import torch.nn as nn

from ..transformer.decoder import DecoderLayer


class Matcher(nn.Module):
    def __init__(self, n_layers, n_heads, h_size, k_size, v_size, dropout=0.):
        super(Matcher, self).__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(n_heads, h_size, k_size, v_size, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input, source, self_mask=None, source_mask=None):
        """
        :param input: An float tensor with shape of [batch_size, input_len, h_size]
        :param source: An float tensor with shape of [batch_size, condition_len, h_size]
        :param self_mask: An byte tensor with shape of [batch_size, input_len]
        :param source_mask: An byte tensor with shape of [batch_size, condition_len]
        :return: An float tensor with shape of [batch_size, input_len, h_size]
        """

        if self_mask is not None:
            self_mask = self_mask.unsqueeze(1).repeat(1, input.size(1), 1)

        if source_mask is not None:
            source_mask = source_mask.unsqueeze(1).repeat(1, input.size(1), 1)

        out = input
        for layer in self.layers:
            out = layer(out, source, self_mask, source_mask)

        return out
