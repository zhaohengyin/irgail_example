import os
import numpy as np
import torch


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.next_actions = tmp['next_action'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)
        self.last_states = tmp['last_state'].clone().to(self.device)
        self.last_actions = tmp['last_action'].clone().to(self.device)
        if 'code' in tmp:
            self.codes = tmp['code'].clone().to(self.device)
        else:
            self.codes = None

        self.prob = np.ones(self.states.size(0)) / self.states.size(0)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        if self.codes is None:
            return (
                self.states[idxes],
                self.actions[idxes],
                self.rewards[idxes],
                self.dones[idxes],
                self.next_states[idxes],
                None
            )
        else:
            return (
                self.states[idxes],
                self.actions[idxes],
                self.rewards[idxes],
                self.dones[idxes],
                self.next_states[idxes],
                self.codes[idxes]
            )

    def sample_full(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.last_actions[idxes],
            self.last_states[idxes],
            self.next_actions[idxes],
            self.next_states[idxes]
        )

    def sample_adaptive(self, batch_size):
        # We need to sample more data from the relevant
        idxes = np.random.choice(self.states.size(0), size=batch_size, p=self.prob)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.last_actions[idxes],
            self.last_states[idxes],
            self.next_actions[idxes],
            self.next_states[idxes]
        )

    def sample_mixture(self, batch_size):
        # We need to sample more data from the relevant
        idxes_0 = np.random.choice(self.states.size(0), size=batch_size // 2, p=self.prob)
        idxes_1 = np.random.randint(low=0, high=self._n, size=batch_size // 2)
        idxes = np.concatenate([idxes_0, idxes_1])
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.last_actions[idxes],
            self.last_states[idxes],
            self.next_actions[idxes],
            self.next_states[idxes]
        )

    def update_prob(self, discriminator):
        probablities = discriminator.calculate_probabilities(self.states).cpu().detach().numpy()
        probablities = probablities.reshape(-1)
        probablities = probablities / np.sum(probablities)
        print(probablities.max(), probablities.min())
       # print(probablities.sum(), self.prob.sum())
        self.prob = 0.9 * probablities + 0.1 * self.prob
        self.prob = self.prob / np.sum(self.prob)
        print(self.prob.max(), self.prob.min())
        #print(self.prob.sum())

class SerializedTransformedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)
        # self.transform = torch.from_numpy(np.array([0.5, 0.2, -0.3, 0.5], np.float32)).to(self.device)

    def transform(self, states):
        a, b = states[:, 0:2], states[:, 2:]
        return torch.cat((a /2, b * 2), dim=1)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        # print(self.states[idxes])

        return (
            self.transform(self.states[idxes]), #+ self.transform,
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.transform(self.next_states[idxes]),# + self.transform
        )


class Buffer(SerializedBuffer):
    def __init__(self, buffer_size, state_shape, action_shape, device, code_shape=0):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)

        self.last_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.last_actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)

        self.next_actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)

        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)

        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

        if code_shape != 0:
            self.codes =  torch.empty(
                (buffer_size, code_shape), dtype=torch.float, device=device)
        else:
            self.codes = None

    def append(self, state, action, reward, done, next_state, next_action=None, last_state=None, last_action=None, code=None):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        if next_action is not None:
            self.next_actions[self._p].copy_(torch.from_numpy(next_action))
        if last_action is not None:
            self.last_actions[self._p].copy_(torch.from_numpy(last_action))
        if last_state is not None:
            self.last_states[self._p].copy_(torch.from_numpy(last_state))
        if code is not None and self.codes is not None:
            self.codes[self._p].copy_(torch.from_numpy(code))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        if self.codes is not None:
            torch.save({
                'state': self.states.clone().cpu(),
                'action': self.actions.clone().cpu(),
                'next_action': self.actions.clone().cpu(),
                'reward': self.rewards.clone().cpu(),
                'done': self.dones.clone().cpu(),
                'next_state': self.next_states.clone().cpu(),
                'last_state' :self.last_states.clone().cpu(),
                'last_action':self.last_actions.clone().cpu(),
                'code': self.codes.clone().cpu()
            }, path)
        else:
            torch.save({
                'state': self.states.clone().cpu(),
                'action': self.actions.clone().cpu(),
                'next_action': self.actions.clone().cpu(),
                'reward': self.rewards.clone().cpu(),
                'done': self.dones.clone().cpu(),
                'next_state': self.next_states.clone().cpu(),
                'last_state': self.last_states.clone().cpu(),
                'last_action': self.last_actions.clone().cpu()
            }, path)



class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        assert self._p % self.buffer_size == 0
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
