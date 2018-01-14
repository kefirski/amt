import torch as t
import torch.nn as nn
from torch.autograd import Variable, grad

from nn.embedding.embedding import Embeddings
from nn.utils import GumbelSoftmax
from . import Translator, Critic


class AMT(nn.Module):
    def __init__(self, path, vocab_size, max_len, pad_idx, layers, heads, h_size, k_size, lambd=50.):
        super(AMT, self).__init__()

        self.pad_idx = pad_idx
        self.lambd = lambd

        # Embeddings do not require gradients – thus we are able to store them even in dict
        self.embeddings = {
            'ru': Embeddings(path['ru'], vocab_size['ru'], max_len['ru'], h_size),
            'en': Embeddings(path['en'], vocab_size['en'], max_len['en'], h_size)
        }

        self.translator = Translator(vocab_size['ru'], layers, heads, h_size, k_size)
        self.critic = Critic(h_size)

    def critic_backward(self, source, input, target):
        """
        :param source: An long tensor with shape of [bs, source_len]
        :param input: An long tensor with shape of [bs, input_len]
        :param target: An long tensor with shape of [bs, input_len]
        """

        bs, _ = source.size()
        cuda = input.is_cuda

        source_mask = t.eq(source, self.pad_idx['en']).data
        target_mask = t.eq(target, self.pad_idx['ru']).data

        source = self.embeddings['en'](source)
        input = self.embeddings['ru'](input)
        target = self.embeddings['ru'](target)

        translation = self.translator(source, input, source_mask)
        bs, sl, vs = translation.size()
        translation = translation.view(-1, vs)
        translation = GumbelSoftmax(translation, 0.96, hard=True).view(bs, sl, vs)
        translation = self.embeddings['ru'](translation)

        source = source.repeat(2, 1, 1)
        source_mask = source_mask.repeat(2, 1)
        target_mask = target_mask.repeat(2, 1)
        critic_input = t.cat([target, translation], 0)

        loss = self.critic(source, critic_input, source_mask, target_mask)

        real_loss, fake_loss = loss[:bs].mean(), loss[bs:].mean()
        wasserstein_distance = -real_loss + fake_loss

        wasserstein_distance.backward()

        eps = t.rand(bs)
        if cuda:
            eps = eps.cuda()
        eps = eps.view(bs, 1, 1).repeat(1, target.size(1), target.size(2))

        interpolation = eps * translation.data + (1 - eps) * target.data
        interpolation = Variable(interpolation, requires_grad=True)

        interpolation_loss = self.critic(source[:bs], interpolation, source_mask[:bs], target_mask[:bs]).mean()
        gradient_penalty = self.gradient_penalty(interpolation, interpolation_loss)

        gradient_penalty.backward()

        return real_loss, fake_loss

    def translator_backward(self, source, input, target):
        """
        :param source: An long tensor with shape of [bs, source_len]
        :param input: An long tensor with shape of [bs, input_len]
        :param target: An long tensor with shape of [bs, input_len]
        """

        bs, _ = source.size()

        source_mask = t.eq(source, self.pad_idx['en']).data
        target_mask = t.eq(target, self.pad_idx['ru']).data
        del target

        source = self.embeddings['en'](source)
        input = self.embeddings['ru'](input)

        translation = self.translator(source, input, source_mask)
        bs, sl, vs = translation.size()
        translation = translation.view(-1, vs)
        translation = GumbelSoftmax(translation, 0.96, hard=True).view(bs, sl, vs)
        translation = self.embeddings['ru'](translation)

        loss = self.critic(source, translation, source_mask, target_mask).mean().neg()

        loss.backward()

        return loss.neg()

    def gradient_penalty(self, interpolation, interpolation_loss):
        gradients = grad(interpolation_loss, interpolation, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return self.lambd * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def critic_train(self):
        self._turn_parameters(self.translator.parameters(), False)
        self._turn_parameters(self.critic.parameters(), True)

    def translator_train(self):
        self._turn_parameters(self.translator.parameters(), True)
        self._turn_parameters(self.critic.parameters(), False)

    @staticmethod
    def _turn_parameters(parameters, value):
        for par in parameters:
            par.requires_grad = value
