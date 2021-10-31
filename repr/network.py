import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from repr.mine import Mine

class ForwardLatentDynamics(nn.Module):
    def __init__(self, zs_dim, za_dim, hidden=(128, 128, 128)):
        super().__init__()
        self.network = build_mlp(input_dim=zs_dim + za_dim,
                                 output_dim=zs_dim,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

    def forward(self, x, y):
        return self.network(torch.cat((x, y), dim=1))

class ForwardLatentDynamicsB(nn.Module):
    def __init__(self, zs_dim, za_dim, hidden=(128,)):
        super().__init__()
        self.network = build_mlp(input_dim=za_dim,
                                 output_dim=zs_dim,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

    def forward(self, x, y):
        return x + self.network(y)

class StateMapping(nn.Module):
    def __init__(self, s_dim, zs_dim, c_dim, hidden=(64, 128, 32)):
        super().__init__()
        self.network = build_mlp(input_dim=s_dim + c_dim,
                                 output_dim=zs_dim,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

    def forward(self, x):
        return self.network(x)

class VariationStateMapping(nn.Module):
    def __init__(self, s_dim, zs_dim, c_dim, cpu=False, hidden=(64, 128, 32)):
        super().__init__()
        self.network = build_mlp(input_dim=s_dim + c_dim,
                                 output_dim=zs_dim * 2,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)
        self.cpu = cpu
        self.zs_dim = zs_dim

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.randn_like(std)
        if not self.cpu:
            eps = eps.cuda()

        return mu + eps * std

    def forward(self, x, no_dec=False):
        stats = self.network(x)
        mu = stats[:, :self.zs_dim]
        logvar = stats[:, self.zs_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            return mu, logvar, z.squeeze()


class StateInverseMapping(nn.Module):
    def __init__(self, s_dim, zs_dim, c_dim, hidden=(64, 128, 32)):
        super().__init__()
        self.network = build_mlp(input_dim=zs_dim + c_dim,
                                 output_dim=s_dim,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

    def forward(self, x):
        return self.network(x)

class InformationEstimator(nn.Module):
    def __init__(self, s_dim, zs_dim, hidden=(64, 128, 32), cpu=False):
        super().__init__()
        self.network = build_mlp(input_dim=s_dim + zs_dim,
                                 output_dim=1,
                                 hidden_activation=nn.ELU(),
                                 hidden_units=hidden)

        self.mine = Mine(self.network, cpu=cpu)

    def mi(self, s, zs):
        return self.mine.mi(s, zs)

    def optimize(self, s, zs):
        return self.mine.optimize_step(s, zs)


class ActionMapping(nn.Module):
    def __init__(self, a_dim, za_dim, c_dim, hidden=(64, 128, 32)):
        super().__init__()
        self.network = build_mlp(input_dim=a_dim + c_dim,
                                 output_dim=za_dim,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

    def forward(self, x):
        return self.network(x)

class VariationalActionMapping(nn.Module):
    def __init__(self, a_dim, za_dim, c_dim, cpu=False, hidden=(64, 128, 32)):
        super().__init__()
        self.network = build_mlp(input_dim=a_dim + c_dim,
                                 output_dim=za_dim * 2,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

        self.cpu = cpu
        self.za_dim = za_dim

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.randn_like(std)
        if not self.cpu:
            eps = eps.cuda()
        return mu + eps * std

    def forward(self, x, no_dec=False):
        stats = self.network(x)
        mu = stats[:, :self.za_dim]
        logvar = stats[:, self.za_dim:]
        z = self.reparametrize(mu, logvar)

        if no_dec:
            return z.squeeze()
        else:
            return mu, logvar, z.squeeze()

class ActionInverseMapping(nn.Module):
    def __init__(self, a_dim, za_dim, c_dim, hidden=(64, 128, 32)):
        super().__init__()
        self.network = build_mlp(input_dim=za_dim + c_dim,
                                 output_dim=a_dim,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

    def forward(self, x):
        return self.network(x)

class Discriminator(nn.Module):
    def __init__(self, zs_dim, hidden=(64, 128, 32)):
        super().__init__()
        self.network = build_mlp(input_dim=zs_dim + zs_dim,
                                 output_dim=1,
                                 hidden_activation=nn.ReLU(),
                                 hidden_units=hidden)

    def forward(self, x):
        return self.network(x)
