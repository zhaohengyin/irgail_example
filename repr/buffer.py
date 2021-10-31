import torch
import numpy as np
from torch_utils import *

class Buffer:
    def __init__(self, buffer_path):
        self.buffer = torch.load(buffer_path)

    def sample(self, batchsize):
        s = []
        a = []
        sn = []
        for env_dict in self.buffer:
            # print(env_dict['s'].shape)
            param = env_dict['p']
            idxes = np.random.randint(low=0, high=env_dict['s'].shape[0], size=batchsize)
            s_batch = numpy_to_tensor(env_dict['s'][idxes])
            a_batch = numpy_to_tensor(env_dict['a'][idxes])
            sn_batch = numpy_to_tensor(env_dict['sn'][idxes])

            param = numpy_to_tensor(param).repeat(batchsize, 1)
            s_batch = torch.cat((s_batch, param), dim=1)
            a_batch = torch.cat((a_batch, param), dim=1)
            sn_batch = torch.cat((sn_batch, param), dim=1)
            s.append(s_batch)
            a.append(a_batch)
            sn.append(sn_batch)

        return torch.cat(s), torch.cat(a), torch.cat(sn)
