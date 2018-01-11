from dataloader import *
from amt import Translator

if __name__ == '__main__':

    loader = Dataloader('./dataloader/data/')
    translator = Translator(loader.vocab_size, loader.max_len, loader.pad_idx, 6, 8, 512, 64)
    source, input, target = loader.torch(3, 'train', False, volatile=True)
    print(target)
    print(translator(source, input))