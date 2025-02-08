import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical
from models.layers.building_blocks import MLP


class HC_Policy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        hc_action_dim: int,
        activation: nn.Module = nn.ReLU(),
    ):
        super(HC_Policy, self).__init__()
        """
        a_dim must be num_options + 1
        """
        # |A| duplicate networks
        self.act = activation

        self._num_weights = hc_action_dim - 1
        self._a_dim = hc_action_dim
        self._dtype = torch.float32

        self.model = MLP(
            input_dim,
            hidden_dim,
            self._a_dim,
            activation=self.act,
            initialization="actor",
        )

    def forward(self, state: torch.Tensor, deterministic=False):
        # when the input is raw by forawrd() not learn()
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
            state = state.reshape(state.shape[0], -1)

        raw_logits = self.model(state)
        logits = F.softplus(raw_logits)

        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        z_argmax = (
            torch.argmax(probs, dim=-1).long()
            if deterministic
            else dist.sample().long()
        )
        z = F.one_hot(z_argmax.long(), num_classes=self._a_dim)

        logprobs = dist.log_prob(z_argmax)
        probs = torch.sum(probs * z, dim=-1)
        entropy = dist.entropy()

        return (
            z,
            z_argmax,
            {"dist": dist, "probs": probs, "logprobs": logprobs, "entropy": entropy},
        )

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions
        logprobs = dist.log_prob(torch.argmax(actions, dim=-1)).unsqueeze(-1)
        return logprobs

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1)


class HC_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.ReLU()
    ):
        super(HC_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation
        self.model = MLP(
            input_dim, hidden_dim, 1, activation=self.act, initialization="critic"
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value, {}


class HC_PPO(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        a_dim: int,
        is_discrete: bool = False,
        activation: nn.Module = nn.Tanh(),
    ):
        super(HC_PPO, self).__init__()
        """
        a_dim must be num_options + 1
        """
        # parameters
        self._a_dim = a_dim
        self.is_discrete = is_discrete

        # |A| duplicate networks
        self.act = activation

        self.model = MLP(
            input_dim, hidden_dim, a_dim, activation=self.act, initialization="actor"
        )

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
            state = state.reshape(state.shape[0], -1)

        raw_logits = self.model(state)

        if self.is_discrete:
            logits = F.softplus(raw_logits)
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                a_argmax = torch.argmax(probs, dim=-1)  # .to(self._dtype)
            else:
                a_argmax = dist.sample()
            a = F.one_hot(a_argmax, num_classes=self._a_dim)

            logprobs = dist.log_prob(a_argmax)
            probs = torch.sum(probs * a, dim=-1)

        else:
            ### Shape the output as desired
            mu = F.tanh(raw_logits)
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)

            covariance_matrix = torch.diag_embed(std**2)  # Variance is std^2
            dist = MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)

            if deterministic:
                a = mu
            else:
                a = dist.rsample()

            logprobs = dist.log_prob(a)
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


class HC_RW(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self, a_dim: int, is_discrete: bool = False, device=torch.device("cpu")
    ):
        super(HC_RW, self).__init__()
        """
        a_dim must be num_options + 1
        """
        # parameters
        self._a_dim = a_dim
        self.is_discrete = is_discrete

        self.device = device
        self.dummy = torch.tensor(1e-10)

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
            state = state.reshape(state.shape[0], -1)

        a = torch.rand((1, self._a_dim)).to(self.device)
        if self.is_discrete:
            a_argmax = torch.argmax(a, dim=1)
            a = F.one_hot(a_argmax, num_classes=self._a_dim)

        dist = torch.zeros_like(state)[:, 0:1]
        probs = torch.zeros_like(dist)
        logprobs = torch.zeros_like(dist)
        entropy = torch.zeros_like(dist)

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

        return torch.zeros_like(dist)

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return torch.zeros_like(dist)
