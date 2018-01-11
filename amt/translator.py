import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from dataloader import *
from nn.transformer import Encoder, Decoder


class Translator(nn.Module):
    def __init__(self, vocab_size, max_len, pad_idx, layers, heads, h_size, k_size):
        """
        :param heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param drop: drop prob
        """
        super(Translator, self).__init__()

        self.vocab_size = vocab_size

        self.encoder = Encoder(vocab_size['en'], max_len['en'], pad_idx['en'], layers, heads, h_size, k_size, k_size)
        self.decoder = Decoder(vocab_size['ru'], max_len['ru'], pad_idx['ru'], layers, heads, h_size, k_size, k_size)

        self.out_fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, self.vocab_size['ru'], bias=False)),
            nn.Softmax(dim=1)
        )

    def forward(self, source, input):
        """
        :param source: An long tensor with shape of [batch_size, condition_len]
        :param input: An long tensor with shape of [batch_size, input_len]
        :return: An float tensor with shape of [batch_size, input_len, vocab_size]
        """

        batch_size, seq_len = input.size()

        source, mask, source_embeddings = self.encoder(source)
        out = self.decoder(input, source, mask)

        out = out.view(batch_size * seq_len, -1)
        out = self.out_fc(out).view(batch_size, seq_len, -1)

        return out, source_embeddings

    def translate(self, source, loader: Dataloader, max_len=80, n_beams=35):

        self.eval()

        use_cuda = source.is_cuda

        source, *_ = self.encoder(source)

        input = loader.go_input(1, use_cuda, lang='ru', volatile=True)

        '''
        Starting point for beam search.
        Generate n_beams tokens
        '''
        out = self.decoder(input, source)
        out = out.view(1, -1)
        out = self.out_fc(out).squeeze(0).data.cpu().numpy()
        beams = Beam.start_search(out, n_beams)

        source = source.repeat(n_beams, 1, 1)

        for _ in range(max_len):

            input = loader.to_tensor([beam.data for beam in beams], use_cuda, lang='ru', volatile=True)

            out = self.decoder(input, source)
            out = out[:, -1]
            out = self.out_fc(out).data.cpu().numpy()

            beams = Beam.update(beams, out)

            '''
            There is no reason to continiue beam search
            if all sequences had already emited stop token
            '''
            if all([any([idx == loader.stop_idx['ru'] for idx in beam.data]) for beam in beams]):
                break

        ''' 
        Actualy, go_idx is the first system token,
        thus in order to check if some idx is beyond sentencepiece vocab, 
        we have to compare this idx with it
        '''
        result = [idx if idx < loader.go_idx['ru'] else 0 for idx in beams[-1].data]
        return ' '.join(map(str, result))

    def learnable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
