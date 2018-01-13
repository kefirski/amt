import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from nn.transformer import Encoder, Matcher


class Critic(nn.Module):
    def __init__(self, layers, heads, h_size, k_size):
        super(Critic, self).__init__()

        self.h_size = h_size

        self.encoder = Encoder(layers, heads, h_size, k_size, k_size)
        self.matcher = Matcher(layers, heads, h_size, k_size, k_size)

        self.out = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.SELU(),

            nn.Linear(h_size, h_size),
            nn.SELU(),

            nn.Linear(h_size, 1)
        )

    def forward(self, source, translation, source_mask=None, translation_mask=None):
        """
        :param source: An float tensor with shape of [batch_size, source_len, h_size]
        :param translation: An float tensor with shape of [batch_size, seq_len, h_size]
        :return: Wass distance estimation with shape of [batch_size]
        """

        source = self.encoder(source, source_mask)
        matching = self.matcher(translation, source, translation_mask, source_mask)

        if translation_mask is not None:
            translation_mask = translation_mask.unsqueeze(2).repeat(1, 1, matching.size(-1))
            matching.data.masked_fill_(translation_mask, 0)

        return self.out(matching.sum(1)).squeeze(1)
