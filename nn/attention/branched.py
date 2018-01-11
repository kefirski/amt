import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal

from nn.transformer.layer_norm import LayerNorm
from .scaled_dot_product import ScaledDotProductAttention


class BranchedAttention(nn.Module):
    def __init__(self, n_heads, h_size, k_size, v_size, p=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param p: drop prob
        """
        super(BranchedAttention, self).__init__()

        self.v_size = v_size

        self.n_heads = n_heads
        self.h_size = h_size

        [self.w_q, self.w_k, self.w_v] = [nn.Parameter(t.FloatTensor(n_heads, h_size, size))
                                          for size in [k_size, k_size, v_size]]
        self.w_o = nn.Parameter(t.FloatTensor(n_heads, v_size, 2 * h_size))

        self.a = nn.Parameter(t.ones(n_heads))
        self.k = nn.Parameter(t.ones(n_heads))

        for param in [self.w_q, self.w_k, self.w_v, self.w_o]:
            xavier_normal(param.data)

        self.out_fc = nn.Sequential(
            nn.Linear(2 * h_size, h_size),
            nn.SELU()
        )

        self.attention = ScaledDotProductAttention(k_size, p)
        self.layer_norm = LayerNorm(h_size)

        self.dropout = nn.Dropout(p)

    def forward(self, q, k, v, mask=None):
        """
        :param q: An float tensor with shape of [batch_size, query_len, hidden_size]
        :param k: An float tensor with shape of [batch_size, seq_len, hidden_size]
        :param v: An float tensor with shape of [batch_size, seq_len, hidden_size]
        :param mask: An byte tensor with shape of [batch_size, query_len, seq_len]
        :return: An float tensor with shape of [batch_size, query_len, hidden_size]
        """

        batch_size = q.size(0)

        q_len = q.size(1)
        seq_len = k.size(1)

        residual = q

        '''
        For now q, k and v are repeated n_heads time 
        and have size of [n_heads, batch_size * len, hidden_size]] each
        '''
        q, k, v = [self.repeat_n_heads(var) for var in [q, k, v]]

        '''
        We project inputs onto corresponding sizes with n_heads independent matrixes.
        After that we perform view over result of this projection in order to obtain
        hidden representations with shape of [batch_size * n_heads, len, hidden_size]
        Note that result have n_heads as dominant size in the first dimention, i.e.
        first batch_size number of batches in result corresponds to first projected head, etc.
        If we would split this tensor with size=batch_size, 
        then we would have array of n_heads length
        with results of mapping each batch through appropriate head 
        '''
        q = self.proj_heads(q, self.w_q, q_len)
        k = self.proj_heads(k, self.w_k, seq_len)
        v = self.proj_heads(v, self.w_v, seq_len)

        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        out, attention = self.attention(q, k, v, mask)
        out = out.view(self.n_heads, -1, self.v_size)
        out = t.bmm(out, self.w_o)

        k = F.softmax(self.k, dim=0)
        out = t.stack([out[i] * k[i] for i in range(self.n_heads)], 0)
        out = out.view(self.n_heads * batch_size, -1, self.h_size * 2)
        out = self.out_fc(out).view(self.n_heads, -1, self.h_size)

        a = F.softmax(self.a, dim=0)
        out = t.stack([out[i] * a[i] for i in range(self.n_heads)], 0)

        out = out.sum(0).view(batch_size, -1, self.h_size)
        out = self.dropout(out)

        return self.layer_norm(out + residual), attention

    def repeat_n_heads(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, hidden_size]
        :return: An float tensor with shape of [n_heads, batch_size * seq_len, hidden_size]
        """

        return input.repeat(self.n_heads, 1, 1).view(self.n_heads, -1, self.h_size)

    @staticmethod
    def proj_heads(input, projection, len):
        """
        :param input: An float tensor with shape of [n_heads, batch_size * len, input_size]
        :param projection: An float tensor with shape of [n_heads, input_size, proj_size]
        :param len: length of input
        :return: An float tensor with shape of [batch_size * n_heads, len, proj_size]
        """

        proj_size = projection.size(2)
        return t.bmm(input, projection).view(-1, len, proj_size)

    def fine_parameters(self):
        return [self.a, self.k]
