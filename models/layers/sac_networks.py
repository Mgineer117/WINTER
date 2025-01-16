import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from models.layers.building_blocks import MLP, Conv, DeConv


class SAC_Policy(nn.Module):
    """
    Soft Actor-Critic (SAC) Policy Network
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        a_dim: int,
        activation: nn.Module = nn.ReLU(),
        is_discrete: bool = False,
    ):
        super(SAC_Policy, self).__init__()

        self.act = activation
        self._a_dim = a_dim
        self._dtype = torch.float32

        self.logstd_range = (-10, 2)

        self.is_discrete = is_discrete

        # Actor network
        if self.is_discrete:
            self.model = MLP(input_dim, hidden_dim, a_dim, activation=self.act)
        else:
            self.model = MLP(input_dim, hidden_dim[:-1], activation=self.act)
            self.mu = MLP(hidden_dim[-1], (a_dim,), activation=nn.Identity())
            self.logstd = MLP(hidden_dim[-1], (a_dim,), activation=nn.Identity())

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        logits = self.model(state)

        if self.is_discrete:
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                a_argmax = torch.argmax(probs, dim=-1)  # .to(self._dtype)
            else:
                a_argmax = dist.sample()
            a = F.one_hot(a_argmax, num_classes=self._a_dim)

            logprobs = dist.log_prob(a_argmax).unsqueeze(-1)
            probs = torch.sum(probs * a, dim=-1)

        else:
            ### Shape the output as desired
            mu = F.tanh(self.mu(logits))
            logstd = torch.clamp(
                self.logstd(logits), min=self.logstd_range[0], max=self.logstd_range[1]
            )
            std = torch.exp(logstd)

            covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            if deterministic:
                a = mu
            else:
                a = dist.rsample()

            logprobs = dist.log_prob(a).unsqueeze(-1)
            probs = torch.exp(logprobs)

        entropy = dist.entropy()

        return a, {
            "dist": dist,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Compute log-probabilities of given actions.
        """
        return dist.log_prob(actions).unsqueeze(-1)

    def entropy(self, dist: torch.distributions):
        """
        Compute entropy of the distribution.
        """
        return dist.entropy().unsqueeze(-1)


class SAC_Critic(nn.Module):
    """
    Soft Actor-Critic (SAC) Critic Network
    Approximates Q-value function Q(s, a).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        activation: nn.Module = nn.ReLU(),
    ):
        super(SAC_Critic, self).__init__()

        self.model = MLP(input_dim, hidden_dim, 1, activation=activation)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # Concatenate state and action along the feature dimension
        x = torch.cat([state, action], dim=-1)
        value = self.model(x)
        return value


class SAC_CriticTwin(nn.Module):
    """
    Twin Critic Network for SAC to mitigate value overestimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        activation: nn.Module = nn.ReLU(),
    ):
        super(SAC_CriticTwin, self).__init__()

        self.critic1 = SAC_Critic(input_dim, hidden_dim, activation)
        self.critic2 = SAC_Critic(input_dim, hidden_dim, activation)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        value1 = self.critic1(state, action)
        value2 = self.critic2(state, action)
        return value1, value2
