import torch as t
import torch.nn as nn
from torch.autograd import Variable, grad

from nn.embedding.embedding import Embeddings
from . import Translator, Critic


class AMT(nn.Module):
    def __init__(self, path, vocab_size, max_len, pad_idx, layers, heads, h_size, k_size, lambd=10.):
        super(AMT, self).__init__()

        self.pad_idx = pad_idx
        self.lambd = lambd

        self.embeddings = {
            'ru': Embeddings(path['ru'], vocab_size['ru'], max_len['ru'], h_size, pad_idx['ru']),
            'en': Embeddings(path['en'], vocab_size['en'], max_len['en'], h_size, pad_idx['en'])
        }

        self.translator = Translator(vocab_size['ru'], layers, heads, h_size, k_size)
        self.critic = Critic(layers, heads, h_size, k_size)

    def train_critic(self, source, input, target, optimizer):
        """
        :param source: An long tensor with shape of [bs, source_len]
        :param input: An long tensor with shape of [bs, input_len]
        :param target: An long tensor with shape of [bs, input_len]
        :param optimizer: Optimizer instance
        """

        bs, _ = source.size()
        cuda = input.is_cuda

        source_mask = t.eq(source, self.pad_idx['en']).data
        target_mask = t.eq(target, self.pad_idx['ru']).data

        source = self.embeddings['en'](source)
        input = self.embeddings['ru'](input)
        target = self.embeddings['ru'](target)

        translation = self.translator(source, input, source_mask)
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
        interpolation = eps * translation.data + (1 - eps) * target.data
        interpolation = Variable(interpolation, requires_grad=True)

        interpolation_loss = self.critic(source[:bs], interpolation, source_mask[:bs], target_mask[:bs]).mean()
        gradient_penalty = self.gradient_penalty(interpolation, interpolation_loss)
        gradient_penalty.backward()

        optimizer.step()

    def train_translator(self, source, input, optimizer):
        """
        :param source: An long tensor with shape of [bs, source_len]
        :param input: An long tensor with shape of [bs, input_len]
        :param optimizer: Optimizer instance
        """

        bs, _ = source.size()

        source_mask = t.eq(source, self.pad_idx['en']).data

        source = self.embeddings['en'](source)
        input = self.embeddings['ru'](input)

        translation = self.translator(source, input, source_mask)
        translation = self.embeddings['ru'](translation)

        loss = -self.critic(source, translation, source_mask).mean()
        loss.backward()
        
        optimizer.step()

    def gradient_penalty(self, interpolation, interpolation_loss):
        gradients = grad(outputs=interpolation_loss, inputs=interpolation,
                         create_graph=True, retain_graph=True, only_inputs=True)[0]
        return self.lambd * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def critic_train(self):
        for parameter in self.translator.parameters():
            parameter.requires_grad = False
        for parameter in self.critic.parameters():
            parameter.requires_grad = True

    def translator_train(self):
        for parameter in self.translator.parameters():
            parameter.requires_grad = True
        for parameter in self.critic.parameters():
            parameter.requires_grad = False
