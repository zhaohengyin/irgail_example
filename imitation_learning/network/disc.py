import torch
from torch import nn
import torch.nn.functional as F
from .utils import build_mlp, cross_entropy_loss
import torch.optim as optim


class StateDiscrim(nn.Module):

    def __init__(self, state_shape, hidden_units=(100, 100),
                 hidden_activation=nn.ReLU()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        ).to(torch.device('cuda'))

    def forward(self, states):
        return self.net(states)

    def calculate_probabilities(self, states):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return F.sigmoid(self.forward(states))

    # def calculate_reward(self, states, next_s):
    #     # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
    #     with torch.no_grad():
    #         probs = self.forward(states, next_states)
    #         # Maximize its loss
    #
    #         return cross_entropy_loss(probs, torch.zeros_like(probs), keepdim=True)

class GAILDiscrim(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            return -F.logsigmoid(-self.forward(states, actions))

