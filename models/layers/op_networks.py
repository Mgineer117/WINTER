import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from models.layers.building_blocks import MLP


class OptionPolicy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        a_dim: int,
        num_weights: int,
        activation: nn.Module = nn.Tanh(),
        is_discrete: bool = False,
    ):
        super(OptionPolicy, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._a_dim = a_dim
        self._dtype = torch.float32

        self.is_discrete = is_discrete
        self.models = nn.ModuleList()
        for _ in range(num_weights):
            self.models.append(self.create_model(input_dim, hidden_dim, a_dim))

    def create_model(self, input_dim, hidden_dim, output_dim):
        return MLP(
            input_dim,
            hidden_dim,
            output_dim,
            activation=self.act,
            initialization="actor",
        )

    def forward(self, state: torch.Tensor, z: int, deterministic=False):
        # when the input is raw by forawrd() not learn()
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        logits = self.models[z](state)

        if self.is_discrete:
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            a_argmax = torch.argmax(probs, dim=-1) if deterministic else dist.sample()
            a = F.one_hot(a_argmax.long(), num_classes=self._a_dim)

            logprobs = dist.log_prob(a_argmax).unsqueeze(-1)
            probs = torch.sum(probs * a, dim=-1)
        else:
            ### Shape the output as desired
            mu = F.tanh(logits)
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)

            covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            a = mu if deterministic else dist.rsample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

        entropy = dist.entropy()

        return a, {
            "z": z,
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions

        if self.is_discrete:
            logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        else:
            logprobs = dist.log_prob(actions).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class OptionCritic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        num_weights: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(OptionCritic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32

        # ex_layer = self.create_sequential_model(sf_dim, fc_dim, 1)

        self.models = nn.ModuleList()
        for _ in range(num_weights):
            self.models.append(self.create_model(input_dim, hidden_dim, 1))

    def create_model(self, input_dim, hidden_dim, output_dim):
        return MLP(
            input_dim,
            hidden_dim,
            output_dim,
            activation=self.act,
            initialization="critic",
        )

    def forward(self, state: torch.Tensor, z: int):
        # when the input is raw by forawrd() not learn()
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)
        value = self.models[z](state)
        return value, {"z": z}


class OP_Q_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        num_weights: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(OP_Q_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._dtype = torch.float32

        # ex_layer = self.create_sequential_model(sf_dim, fc_dim, 1)

        self.models = nn.ModuleList()
        for _ in range(num_weights):
            self.models.append(self.create_model(input_dim, hidden_dim, 1))

    def create_model(self, input_dim, hidden_dim, output_dim):
        return MLP(input_dim, hidden_dim, output_dim, activation=self.act)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: int):
        x = torch.cat([states, actions], dim=-1)
        value = self.models[z](x)
        return value


class OPtionCriticTwin(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        num_weights: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(OPtionCriticTwin, self).__init__()

        # |A| duplicate networks
        self._dtype = torch.float32

        self.critic1 = OP_Q_Critic(input_dim, hidden_dim, num_weights, activation)
        self.critic2 = OP_Q_Critic(input_dim, hidden_dim, num_weights, activation)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: int):
        value1 = self.critic1(states, actions, z)
        value2 = self.critic2(states, actions, z)
        return value1, value2, {"z": z}
