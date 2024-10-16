import torch
import torch.nn as nn
import numpy as np
from .layer_tools import construct_default_layer
from torch.distributions import Normal


class Agent(nn.Module):
    def __init__(
        self,
        envs,
        init_log_std=0.0,
        n_hidden_neurons=64,
        origin=None,
        critic_infinitesimal=False,
        pos_def_penalty_coeff=None,
    ):
        super().__init__()
        self.n_hidden_neurons = n_hidden_neurons
        self.init_log_std = init_log_std
        self.envs = envs
        self.critic_infinitesimal = critic_infinitesimal
        self.origin = origin
        self.out_layer = construct_default_layer(
            64, 1, std=1.0, bias=not critic_infinitesimal
        )
        self.n_observation_space_components: int = np.array(
            envs.single_observation_space.shape
        ).prod()
        self.n_action_space_components = np.array(envs.single_action_space.shape).prod()
        self.pos_def_penalty_coeff = pos_def_penalty_coeff

        self.critic = nn.Sequential(
            construct_default_layer(
                self.n_observation_space_components, 64, bias=not critic_infinitesimal
            ),
            nn.Tanh(),
            construct_default_layer(64, 64, bias=not critic_infinitesimal),
            nn.Tanh(),
            self.out_layer,
        )
        self.actor_mean = nn.Sequential(
            construct_default_layer(self.n_observation_space_components, 64),
            nn.Tanh(),
            construct_default_layer(64, 64),
            nn.Tanh(),
            construct_default_layer(64, self.n_action_space_components, std=0.01),
        )
        self.actor_features = nn.Sequential(
            construct_default_layer(self.n_observation_space_components, 64),
            nn.Tanh(),
            construct_default_layer(64, 64),
            nn.Tanh(),
        )
        self.actor_action = nn.Linear(64, self.n_action_space_components, bias=False)
        self.actor_logstd = nn.Parameter(
            self.init_log_std * torch.ones(1, self.n_action_space_components)
        )

    def get_value(self, x):
        if self.critic_infinitesimal and self.origin is not None:
            x = x - self.origin

        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        actor_features = self.actor_features(x)
        action_mean = self.actor_action(actor_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.get_value(x),
        )


class AgentSDE(Agent):
    def __init__(
        self,
        envs,
        init_log_std=0.0,
        n_hidden_neurons=64,
        origin=None,
        critic_infinitesimal=False,
        pos_def_penalty_coeff=None,
    ):
        super().__init__(
            envs,
            init_log_std,
            n_hidden_neurons,
            origin=origin,
            critic_infinitesimal=critic_infinitesimal,
            pos_def_penalty_coeff=pos_def_penalty_coeff,
        )
        self.actor_logstd = nn.Parameter(
            self.init_log_std
            * torch.ones(64, np.prod(self.envs.single_action_space.shape))
        )
        self.exploration_matrix = torch.FloatTensor(
            torch.zeros(64, self.n_action_space_components)
        )

    def reset_noise(self):
        self.exploration_matrix = Normal(loc=0, scale=self.actor_logstd.exp()).sample()

    def get_action_and_value(self, x, action=None):
        actor_features = self.actor_features(x)
        action_mean = self.actor_action(actor_features)
        if action is None:
            action_std = actor_features @ self.exploration_matrix
            action = action_mean + action_std

        probs = Normal(
            action_mean,
            torch.sqrt(
                1e-6 + torch.mm(actor_features**2, self.actor_logstd.exp() ** 2)
            ),
        )
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.get_value(x),
        )


# class AgentVF(Agent):
#
#     def __init__(self, envs, init_log_std=0.0, n_hidden_neurons=64, origin=None):
#         super().__init__(
#             envs=envs, init_log_std=init_log_std, n_hidden_neurons=n_hidden_neurons
#         )
#         self.n_hidden_neurons = n_hidden_neurons
#         self.init_log_std = init_log_std
#         self.envs = envs
#         self.out_layer = construct_default_layer(64, 1, std=1.0)
#
#         n_observation_space_components: int = np.array(
#             envs.single_observation_space.shape
#         ).prod()
#         n_action_space_components = np.array(envs.single_action_space.shape).prod()
#
#         self.critic = nn.Sequential(
#             construct_default_layer(n_observation_space_components, 64, bias=False),
#             nn.Tanh(),
#             construct_default_layer(64, 64, bias=False),
#             nn.Tanh(),
#             self.out_layer,
#         )
#         self.actor_mean = nn.Sequential(
#             construct_default_layer(n_observation_space_components, 64),
#             nn.Tanh(),
#             construct_default_layer(64, 64),
#             nn.Tanh(),
#             construct_default_layer(64, n_action_space_components, std=0.01),
#         )
#         self.actor_features = nn.Sequential(
#             construct_default_layer(n_observation_space_components, 64),
#             nn.Tanh(),
#             construct_default_layer(64, 64),
#             nn.Tanh(),
#         )
#         self.actor_action = nn.Linear(64, n_action_space_components, bias=False)
#         self.actor_logstd = nn.Parameter(
#             self.init_log_std * torch.ones(1, n_action_space_components)
#         )
#         self.exploration_matrix = torch.FloatTensor(
#             torch.zeros(64, n_action_space_components)
#         )
#         self.origin = origin
#     def soft_abs(self, x):
#         return torch.log(1 + torch.abs(x))
#
#     def get_value(self, x):
#         bias_tensor = self.origin if self.origin is not None else torch.zeros_like(x)
#         critic_diff = self.critic(x - bias_tensor)
#         # l2_norm_diff = torch.linalg.norm(x - bias_tensor, dim=1, ord=2, keepdim=True)
#         # value = -self.soft_abs(critic_diff) - (l2_norm_diff * 0.01)
#         return critic_diff
