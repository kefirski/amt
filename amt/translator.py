import torch.nn as nn
from torch.nn.utils import weight_norm

from dataloader import *
from nn.transformer import Encoder, Decoder


class Translator(nn.Module):
    def __init__(self, vocab_size, layers, heads, h_size, k_size):
        super(Translator, self).__init__()

        self.vocab_size = vocab_size

        self.encoder = Encoder(layers, heads, h_size, k_size, k_size)
        self.decoder = Decoder(layers, heads, h_size, k_size, k_size)

        self.out_fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, self.vocab_size, bias=False)),
            nn.Softmax(dim=1)
        )

    def forward(self, source, input, mask=None):
        """
        :param source: An float tensor with shape of [batch_size, condition_len, h_size]
        :param input: An float tensor with shape of [batch_size, input_len, h_size]
        :return: An float tensor with shape of [batch_size, input_len, vocab_size]
        """

        batch_size, seq_len, _ = input.size()

        source = self.encoder(source, mask)
        out = self.decoder(input, source, mask)

        out = out.view(batch_size * seq_len, -1)
        out = self.out_fc(out).view(batch_size, seq_len, -1)

        return out

    def translate(self, source, embeddings, loader: Dataloader, max_len=80, n_beams=35):

        self.eval()

        use_cuda = source.is_cuda

        source = embeddings['en'](source)
        source = self.encoder(source)

        input = loader.go_input(1, use_cuda, lang='ru', volatile=True)
        input = embeddings['ru'](input)

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
            input = embeddings['ru'](input)

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
