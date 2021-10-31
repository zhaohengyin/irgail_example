import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch_utils
from repr.network import *
from utils import *


def kl_divergence(mu, logvar):
    kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
    return kld


class RepresentationNetwork:
    def __init__(self, s_dim, a_dim, c_dim, zs_dim, za_dim, buffer,
                 h_inform=(64, 128, 32),  h_forward=(), h_sinv=(128, 128, 128),
                 c_recon=1, c_kl=0.1, c_inv=1, c_f=2.0, c_conf=300, name='none', device=torch.device('cuda')):

        self.latent_forward_model = ForwardLatentDynamics(zs_dim, za_dim, hidden=h_forward).to(device)

        self.action_inverse_mapping = ActionInverseMapping(a_dim, za_dim, c_dim).to(device)

        self.state_inverse_mapping = StateInverseMapping(s_dim, zs_dim, c_dim, h_sinv).to(device)
        # self.latent_discrim = Discriminator(zs_dim)
        # self.information = InformationEstimator(s_dim, zs_dim).to(torch.device('cuda'))
        if device == torch.device('cpu'):
            self.state_mapping = VariationStateMapping(s_dim, zs_dim, c_dim, cpu=True).to(device)
            self.action_mapping = VariationalActionMapping(a_dim, za_dim, c_dim, cpu=True).to(device)

            self.confusion_estimator_s = InformationEstimator(zs_dim, c_dim, hidden=h_inform, cpu=True).to(device)
            self.confusion_estimator_a = InformationEstimator(za_dim, c_dim, hidden=h_inform, cpu=True).to(device)
        else:
            self.state_mapping = VariationStateMapping(s_dim, zs_dim, c_dim).to(device)
            self.action_mapping = VariationalActionMapping(a_dim, za_dim, c_dim).to(device)

            self.confusion_estimator_s = InformationEstimator(zs_dim, c_dim, hidden=h_inform).to(device)
            self.confusion_estimator_a = InformationEstimator(za_dim, c_dim, hidden=h_inform).to(device)

        self.main_networks = {'lf': self.latent_forward_model,
                              'af': self.action_mapping,
                              'aif': self.action_inverse_mapping,
                              'sf': self.state_mapping,
                              'sif': self.state_inverse_mapping}

        self.forward_optimizer = optim.Adam(list(self.latent_forward_model.parameters()), lr=0.0003, weight_decay=0.001)

        self.generator_optimizer = optim.Adam(list(self.action_mapping.parameters())
                                              + list(self.action_inverse_mapping.parameters())
                                              + list(self.state_mapping.parameters())
                                              + list(self.state_inverse_mapping.parameters()), lr=0.0003)

        self.main_optimizer = optim.Adam(list(self.latent_forward_model.parameters())
                                         + list(self.action_mapping.parameters())
                                         + list(self.action_inverse_mapping.parameters())
                                         + list(self.state_mapping.parameters())
                                         + list(self.state_inverse_mapping.parameters()), lr=0.0003)
        self.c_kl = c_kl
        self.c_recon = c_recon
        self.c_inv = c_inv

        self.c_f = c_f
        self.c_conf = c_conf

        self.c_dim = c_dim
        self.buffer = buffer
        self.name = name

    def save(self, path):
        param = {}
        for k in self.main_networks:
            param[k] = self.main_networks[k].state_dict()
        param['cons'] = self.confusion_estimator_s.state_dict()
        param['cona'] = self.confusion_estimator_a.state_dict()
        torch.save(param, path)

    def load(self, path, cpu=False):
        if cpu:
            params = torch.load(path, map_location=torch.device('cpu'))
        else:
            params = torch.load(path)

        # params = torch.load(path)
        for k in self.main_networks:
            self.main_networks[k].load_state_dict(params[k])
        self.confusion_estimator_s.load_state_dict(params['cons'])
        self.confusion_estimator_a.load_state_dict(params['cona'])

    def encode_action(self, action, code, repeat=True):
        if repeat:
            codes = torch_utils.numpy_to_tensor(code).reshape(-1).repeat(action.size(0), 1)
        else:
            codes= code
        return self.action_mapping(torch.cat((action, codes), dim=1))[-1]

    def encode_state(self, state, code, repeat=True):
        if repeat:
            codes = torch_utils.numpy_to_tensor(code).reshape(-1).repeat(state.size(0), 1)
        else:
            codes = code
        return self.state_mapping(torch.cat((state, codes), dim=1))[-1]

    def train(self, itrs, itr_steps, save_path):
        total_step = 0
        for itr in range(itrs):
            for step in range(itr_steps):
                s, a, sn = self.buffer.sample(24)

                mu, logvar, zs = self.state_mapping(s)
                mu_next, _, zsn = self.state_mapping(sn)
                # _, _, zsn = self.state_mapping(sn)
                mua, logvara, za = self.action_mapping(a)

                zsn_predict = self.latent_forward_model(zs, za)

                # Dynamic consistency
                # sn_recover_f = self.state_inverse_mapping(torch.cat((zsn_predict, sn[:, -self.c_dim:]), dim=-1))
                loss_latent_forward = self.c_f * torch.sum(
                    0.5 * (zsn - zsn_predict) * (zsn - zsn_predict), dim=1).mean()


                # Mutual Information consistency
                s_recover = self.state_inverse_mapping(torch.cat((zs, s[:, -self.c_dim:]), dim=-1))
                sn_recover = self.state_inverse_mapping(torch.cat((zsn, sn[:, -self.c_dim:]), dim=-1))

                # print(s_recover - s[:, :-self.c_dim])

                loss_state = self.c_recon * (
                            torch.sum(0.5 * (s_recover - s[:, :-self.c_dim]) * (s_recover - s[:, :-self.c_dim]),
                                      dim=1).mean()
                            + torch.sum(0.5 * (sn_recover - sn[:, :-self.c_dim]) * (sn_recover - sn[:, :-self.c_dim]),
                                        dim=1).mean())

                # Action Inverse Loss.
                za_augmented = torch.cat((za, a[:, -self.c_dim:]), dim=1)
                action = self.action_inverse_mapping(za_augmented)

                loss_action = torch.sum(0.5 * (a[:, :-self.c_dim] - action) * (a[:, :-self.c_dim] - action),
                                        dim=1).mean()

                # Domain Confusion Loss
                loss_confusion = (self.confusion_estimator_s.mi(zs, s[:, -self.c_dim:]) +
                                  self.confusion_estimator_a.mi(za, a[:, -self.c_dim:])) * self.c_conf

                # Loss KL:
                loss_kl = (kl_divergence(mu, logvar) + kl_divergence(mua, logvara)) * self.c_kl

                # Total.

                print('Step:{}| Forward: {}, State: {}, Action: {}，Confusion: {}, KL: {}'.
                      format(total_step, loss_latent_forward, loss_state, loss_action, loss_confusion, loss_kl))

                total_step += 1

                loss = loss_latent_forward + loss_state + loss_action + loss_confusion  + loss_kl

                self.forward_optimizer.zero_grad()
                self.generator_optimizer.zero_grad()
                loss.backward()
                self.forward_optimizer.step()
                self.generator_optimizer.step()

                # Train Information Estimator.
                for i in range(4):
                    s, a, sn = self.buffer.sample(2)
                    # sn_flatten = dict_config_concat(sn)
                    # a_flatten = dict_config_concat(a)

                    _, _, zs = self.state_mapping(s)
                    _, _, za = self.action_mapping(a)
                    # zsn = self.state_mapping(sn_flatten)
                    # za = self.action_mapping(a_flatten)
                    self.confusion_estimator_s.optimize(zs, s[:, -self.c_dim:])
                    self.confusion_estimator_a.optimize(za, a[:, -self.c_dim:])
                    # self.information.optimize(s[:, :-self.c_dim], zs)

            import os
            epoch_model_save_path = os.path.join(save_path, '{}_itr_{}.pth'.format(self.name, itr))
            self.save(epoch_model_save_path)
