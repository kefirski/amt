from math import sqrt

import numpy as np
import torch as t
import torch.nn as nn
from gensim.models.keyedvectors import KeyedVectors
from torch.autograd import Variable


class Embeddings(nn.Module):
    def __init__(self, path, vocab_size, max_len, h_size):
        super(Embeddings, self).__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.h_size = h_size

        self.token_embeddings = nn.Embedding(vocab_size, h_size)
        self.positional_embeddings = nn.Embedding(int(max_len), h_size, padding_idx=0)

        self._token_embedding_init(path)
        self._position_embedding_init()

    def forward(self, input):

        mixed = len(input.size()) == 3

        batch_size, seq_len, *_ = input.size()

        positional = Variable(t.LongTensor([i for i in range(1, seq_len + 1)])).repeat(batch_size).view(batch_size, -1)
        if input.is_cuda:
            positional = positional.cuda()

        if mixed:
            input = input.view(batch_size * seq_len, -1)
            return t.mm(input, self.token_embeddings.weight).view(batch_size, seq_len, -1) + \
                   self.positional_embeddings(positional)
        else:
            return self.token_embeddings(input) + self.positional_embeddings(positional)

    def _randn_embed(self):
        return np.random.randn(self.h_size) / sqrt(self.h_size)

    def _token_embedding_init(self, path):
        """
        :param path: Path to pretrained embeddings for each index in vocabulary
        """
        keyed_vectors = KeyedVectors.load_word2vec_format(path, binary=True)

        embeddings = np.array([keyed_vectors.wv[str(idx)] if str(idx) in keyed_vectors.vocab else self._randn_embed()
                               for idx in range(self.vocab_size)])

        self.token_embeddings.weight = nn.Parameter(t.from_numpy(embeddings).float(), requires_grad=False)

    def _position_embedding_init(self):
        encoding = np.array([
            [pos / np.power(10000, 2 * i / self.h_size) for i in range(self.h_size)]
            if pos != 0 else np.zeros(self.h_size) for pos in range(self.max_len)
        ])

        encoding[1:, 0::2] = np.sin(encoding[1:, 0::2])
        encoding[1:, 1::2] = np.cos(encoding[1:, 1::2])

        self.positional_embeddings.weight = nn.Parameter(t.from_numpy(encoding).float(), requires_grad=False)
