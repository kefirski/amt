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
            weight_norm(nn.Linear(h_size, h_size)),
            nn.SELU(),

            weight_norm(nn.Linear(h_size, h_size)),
            nn.SELU(),

            weight_norm(nn.Linear(h_size, h_size / 2)),
            nn.SELU(),

            weight_norm(nn.Linear(h_size / 2, 200)),
            nn.SELU(),

            weight_norm(nn.Linear(200, 50)),
            nn.SELU(),

            weight_norm(nn.Linear(50, 1))
        )

    def forward(self, source, translation, source_mask=None, translation_mask=None):
        """
        :param source: An float tensor with shape of [batch_size, source_len, h_size]
        :param translation: An float tensor with shape of [batch_size, seq_len, h_size]
        :return: Wass distance estimation with shape of [batch_size]
        """

        source = self.encoder(source, source_mask)
        matching = self.matcher(translation, source, translation_mask, source_mask)

        matching = matching.sum(1)
        return self.out(matching).squeeze(1)
