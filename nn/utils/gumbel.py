import torch as t
import torch.nn.functional as F
from torch.autograd import Variable


def GumbelSoftmax(input, temperatute, hard=False):
    """
    :param input: An float tensor with shape of [batch_size, input_size]
    :param temperatute: Non-negative scalar
    :param hard: if True, take argmax, but differentiate w.r.t. soft sample y
    :return: An float tensor with shape of [batch_size, input_size]
    """

    batch_size, input_size = input.size()

    noise = Variable(t.rand(batch_size, input_size))
    noise = -t.log(-t.log(noise + 1e-20) + 1e-20)
    if input.is_cuda:
        noise = noise.cuda()

    result = F.softmax((input + noise) / temperatute, dim=1)

    if hard:
        hard_result = result == t.max(result, 1)[0].unsqueeze(1).repeat(1, input_size)
        hard_result = hard_result.float()

        result = (hard_result - result).detach() + result

    return result
