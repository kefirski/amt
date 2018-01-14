import torch as t
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, h_size):
        super(Critic, self).__init__()

        self.h_size = h_size

        self.net = nn.Sequential(
            nn.Conv2d(2 * h_size, 2 * h_size, 5, padding=1, bias=True),
            nn.SELU(),

            nn.Conv2d(2 * h_size, h_size, 5, padding=1, bias=True),
            nn.SELU(),

            nn.Conv2d(h_size, h_size, 5, padding=1, bias=True),
            nn.SELU(),

            nn.Conv2d(h_size, h_size, 5, padding=1, bias=True),
            nn.SELU(),

            nn.Conv2d(h_size, 1, 5, padding=1, bias=True)
        )

    def forward(self, source, translation, source_mask=None, translation_mask=None):
        """
        :param source: An float tensor with shape of [batch_size, source_len, h_size]
        :param translation: An float tensor with shape of [batch_size, seq_len, h_size]
        :return: Wass distance estimation with shape of [batch_size]
        """

        if translation_mask is not None:
            translation_mask = translation_mask.unsqueeze(2).repeat(1, 1, self.h_size)
            translation.data.masked_fill_(translation_mask, 0)

        if source_mask is not None:
            source_mask = source_mask.unsqueeze(2).repeat(1, 1, self.h_size)
            source.data.masked_fill_(source_mask, 0)

        source_len = source.size(1)
        translation_len = translation.size(1)

        source = source.unsqueeze(1).repeat(1, translation_len, 1, 1)
        translation = translation.unsqueeze(1).transpose(1, 2).repeat(1, 1, source_len, 1)

        input = t.cat([source, translation], 3).transpose(1, 3)

        return self.net(input).squeeze(1).sum(1).sum(1)
