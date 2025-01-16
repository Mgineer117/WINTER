import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from models.layers.building_blocks import MLP, Conv, DeConv


class PPO_Policy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        a_dim: int,
        activation: nn.Module = nn.Tanh(),
        is_discrete: bool = False,
    ):
        super(PPO_Policy, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._a_dim = a_dim
        self._dtype = torch.float32

        self.logstd_range = (-10, 2)

        self.is_discrete = is_discrete

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


class PPO_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super(PPO_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation
        self._dtype = torch.float32

        self.model = MLP(input_dim, hidden_dim, 1, activation=self.act)

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
