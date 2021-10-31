import torch
import torch.nn
import numpy as np
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


def move_to_gpu(var):
    if use_cuda:
        return var.cuda()
    else:
        return var


def numpy_to_tensor(var):
    return move_to_gpu(Variable(torch.FloatTensor(var)))


def scalar_to_tensor(var):
    return move_to_gpu(Variable(torch.FloatTensor(np.array([var]))))


def tensor_to_scalar(t):
    return tensor_to_numpy(t).reshape(-1)[0]


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def tensor_normalize(tensor):
    if len(tensor.shape) == 2:
        return tensor / torch.norm(tensor, p=2, dim=1)
    else:
        return tensor / torch.norm(tensor, p=2, dim=0).reshape(-1)

