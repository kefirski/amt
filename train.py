import argparse
import subprocess
from nn.utils import ScheduledOptim

import torch as t
from torch.optim import Adam

from amt import AMT
from dataloader import Dataloader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='inf')
    parser.add_argument('--num-iterations', type=int, default=1_500_000, metavar='NI',
                        help='num iterations (default: 1_500_000)')
    parser.add_argument('--batch-size', type=int, default=20, metavar='BS',
                        help='batch size (default: 20)')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--lambd', type=float, default=10., metavar='LA',
                        help='lambda value (default: 10.)')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='D',
                        help='dropout rate (default: 0.1)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    args = parser.parse_args()

    t.set_num_threads(args.num_threads)
    loader = Dataloader('./dataloader/data/')

    amt = AMT({'en': './dataloader/data/en_embeddings.bin', 'ru': './dataloader/data/ru_embeddings.bin'},
              loader.vocab_size, loader.max_len, loader.pad_idx,
              layers=6, heads=8, h_size=512, k_size=64, lambd=args.lambd)
    if args.use_cuda:
        amt = amt.cuda()
        for embed in amt.embeddings.values():
            embed = embed.cuda()

    translator_optim = ScheduledOptim(Adam(amt.translator.parameters(), betas=(0.9, 0.98), eps=1e-9), 512, 4000)
    critic_optim = ScheduledOptim(Adam(amt.critic.parameters(), betas=(0.9, 0.98), eps=1e-9), 512, 1000)

    print('Model have initialized')

    for i in range(args.num_iterations):

        amt.critic_train()

        for j in range(1):
            critic_optim.zero_grad()
            translator_optim.zero_grad()
            for k in range(8):
                source, input, target = loader.torch(args.batch_size, 'train', args.use_cuda)

                real_loss, fake_loss = amt.critic_backward(source, input, target)
            if i % 10 == 0:
                print('critic i {} j {} real {} fake {} wass {}'.format(i, j,
                                                                        real_loss.cpu().data.numpy()[0],
                                                                        fake_loss.cpu().data.numpy()[0],
                                                                        (real_loss - fake_loss).cpu().data.numpy()[0]))

            critic_optim.step()

        amt.translator_train()

        critic_optim.zero_grad()
        translator_optim.zero_grad()
        for j in range(8):
            source, input, target = loader.torch(args.batch_size, 'train', args.use_cuda)

            loss = amt.translator_backward(source, input, target)

        if i % 10 == 0:
            print('translator i {} fake {}'.format(i, loss.cpu().data.numpy()[0]))

        translator_optim.step()

        if i % 300 == 0:
            source, _, target = loader.torch(1, 'valid', args.use_cuda, volatile=True)
            indexes = ' '.join(map(str, source[0].cpu().data.numpy()[1:-1]))
            subprocess.Popen(
                'echo "{}" | spm_decode --model=./dataloader/data/en.model --input_format=id'.format(indexes),
                shell=True
            )
            print('_________')
            indexes = ' '.join(map(str, target[0].cpu().data.numpy()[1:-1]))
            subprocess.Popen(
                'echo "{}" | spm_decode --model=./dataloader/data/ru.model --input_format=id'.format(indexes),
                shell=True
            )
            print('_________')
            indexes = amt.translator.translate(source, amt.embeddings, loader, max_len=60, n_beams=12)
            subprocess.Popen(
                'echo "{}" | spm_decode --model=./dataloader/data/ru.model --input_format=id'.format(indexes),
                shell=True
            )
            print('_________')
