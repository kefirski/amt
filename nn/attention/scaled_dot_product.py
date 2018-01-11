from math import sqrt

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, size, p=0.1):
        """
        :param size: float number that is necessary for estimation scaling factor
        :param m_size: int number of size of the window that performing local-m attention.
                   None corresponds to global attention mechanism
        :param p: drop prob
        """
        super(ScaledDotProductAttention, self).__init__()

        self.scaling = 1 / (sqrt(size))
        self.dropout = nn.Dropout(p)

    def forward(self, q, k, v, mask=None):
        """
        :param q: An float tensor with shape of [batch_size, query_len, size]
        :param k: An float tensor with shape of [batch_size, seq_len, size]
        :param v: An float tensor with shape of [batch_size, seq_len, value_size]
        :param mask: An byte tensor with shape of [batch_size, query_len, seq_len]
        :return: An float tensor with shape of [batch_size, query_len, value_size]
                     and attention map with shape of [batch_size, query_len, seq_len]
        """

        batch_size, query_len, _ = q.size()

        attention = t.bmm(q, k.transpose(1, 2)) * self.scaling

        '''
        In order to prevent contribution of padding symbols in attention lockup, 
        it is necessary to use attention mask
        '''
        if mask is not None:
            attention.data.masked_fill_(mask, -float('inf'))

        attention = F.softmax(attention, dim=2)

        return t.bmm(self.dropout(attention), v), attention
