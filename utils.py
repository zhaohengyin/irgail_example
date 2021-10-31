from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

def build_mlp(input_dim, output_dim, hidden_units=[64, 64],
              hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)

def dict_concat(x):
    return torch.cat([value for key, value in x.items()], dim=0)

def dict_config_concat(x):
    return torch.cat([torch.cat((value, key.repeat(value.size(0),1)), dim=1) for key, value in x.items()], dim=0)
