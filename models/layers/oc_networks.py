import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Categorical, Bernoulli
from torch.nn import MaxPool2d, MaxUnpool2d
from models.layers.building_blocks import MLP, Conv, DeConv


class OC_Policy(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        state_dim: int,
        termination_dim: list,
        fc_dim: int,
        a_dim: int,
        num_options: int,
        temperature: float = 1.0,
        eps_start: float = 1.0,
        eps_min: float = 0.1,
        eps_decay: int = int(5e5),
        eps_test: float = 0.05,
        activation: nn.Module = nn.Tanh(),
        is_discrete: bool = False,
    ):
        super(OC_Policy, self).__init__()

        # |A| duplicate networks
        self.act = activation

        self._a_dim = a_dim
        self._num_options = num_options

        self._temperature = temperature
        self._eps_min = eps_min
        self._eps_start = eps_start
        self._eps_decay = eps_decay
        self._eps_test = eps_test
        self._num_steps = 0

        self._dtype = torch.float32

        self.logstd_range = (-10, 2)

        self.is_discrete = is_discrete

        # Option-Termination
        self.terminations = MLP(
            state_dim, termination_dim, num_options, activation=nn.Tanh()
        )

        if self.is_discrete:
            self.option_W = nn.Parameter(
                torch.zeros(self._num_options, state_dim, self._a_dim)
            )
            self.option_b = nn.Parameter(torch.zeros(self._num_options, self._a_dim))
        else:
            self.option_W = nn.Parameter(
                torch.zeros(self._num_options, state_dim, fc_dim)
            )
            self.option_b = nn.Parameter(torch.zeros(self._num_options, fc_dim))

            self.mu = MLP(fc_dim, (a_dim,), activation=nn.Identity())
            self.logstd = MLP(fc_dim, (a_dim,), activation=nn.Identity())

    def get_terminations(self, state: torch.Tensor):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        return F.sigmoid(self.terminations(state))

    def predict_option_termination(self, state: torch.Tensor, z: int):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        termination = F.sigmoid(self.terminations(state)[:, z])
        option_termination = Bernoulli(termination).sample()
        return bool(option_termination.item())  # , next_option.item()

    def epsilon(self, is_eval: bool = True):
        if is_eval:
            eps = self._eps_test
        else:
            eps = self._eps_min + (self._eps_start - self._eps_min) * math.exp(
                -self._num_steps / self._eps_decay
            )
            self._num_steps += 1
        return eps

    def forward(self, state: torch.Tensor, z: int, deterministic: bool = False):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        logits = state.data @ self.option_W[z] + self.option_b[z]

        if self.is_discrete:
            probs = F.softmax(logits / self._temperature, dim=-1)
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


class OC_Critic(nn.Module):
    """
    Psi Advantage Function: Psi(s,a) - (1/|A|)SUM_a' Psi(s, a')
    """

    def __init__(
        self,
        input_dim: int,
        fc_dim: int,
        num_options: int,
        activation: nn.Module = nn.Tanh(),
    ):
        super(OC_Critic, self).__init__()

        # |A| duplicate networks
        self.act = activation
        self._dtype = torch.float32

        self.model = MLP(input_dim, (fc_dim, fc_dim), num_options, activation=self.act)

    def forward(self, state: torch.Tensor):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        Q = self.model(state)
        return Q

    def greedy_option(self, state: torch.Tensor):
        if len(state.shape) == 3 or len(state.shape) == 1:
            state = state.unsqueeze(0)
        state = state.reshape(state.shape[0], -1)

        Q = self.model(state)
        return torch.argmax(Q, dim=-1, keepdim=True)
