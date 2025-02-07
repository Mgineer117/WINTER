import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b as bfgs

from copy import deepcopy
from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils import estimate_advantages
from models.layers.building_blocks import MLP
from models.layers.ppo_networks import PPO_Policy, PPO_Critic
from models.policy.base_policy import BasePolicy


class PPO_Learner(BasePolicy):
    def __init__(
        self,
        policy: PPO_Policy,
        critic: PPO_Critic,
        policy_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        minibatch_size: int = 256,
        eps: float = 0.2,
        entropy_scaler: float = 1e-3,
        l2_reg: float = 1e-5,
        target_kl: float = 0.03,
        gamma: float = 0.99,
        gae: float = 0.9,
        K: int = 5,
        device: str = "cpu",
    ):
        super(PPO_Learner, self).__init__()

        # constants
        self.device = device

        self.minibatch_size = minibatch_size
        self._entropy_scaler = entropy_scaler
        self._eps = eps
        self._gamma = gamma
        self._gae = gae
        self._K = K
        self._l2_reg = l2_reg
        self._target_kl = target_kl
        self._forward_steps = 0

        # trainable networks
        self.policy = policy
        self.critic = critic

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": policy_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        #
        self.dummy = torch.tensor(0.0)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def preprocess_obs(self, obs):
        observation = obs["observation"]

        # preprocessing
        observation = torch.from_numpy(observation).to(self._dtype).to(self.device)

        return {"observation": observation}

    def forward(self, obs, z=None, deterministic=False):
        """
        Image-based state dimension ~ [Batch, width, height, channel] or [width, height, channel]
        Flat tensor-based state dimension ~ [Batch, tensor] or [tensor]
        z is dummy input for code consistency
        """
        self._forward_steps += 1
        obs = self.preprocess_obs(obs)

        a, metaData = self.policy(obs["observation"], deterministic=deterministic)

        return a, {
            # "z": self.dummy.item(),
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def learn(self, batch, z=0):
        """Performs a single training step using PPO, incorporating all reference training steps."""
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"]).reshape(batch["states"].shape[0], -1)
        actions = to_tensor(batch["actions"])
        rewards = to_tensor(batch["rewards"])
        terminals = to_tensor(batch["terminals"])
        old_logprobs = to_tensor(batch["logprobs"])

        # Compute advantages and returns
        with torch.no_grad():
            values = self.critic(states)
            advantages, returns = estimate_advantages(
                rewards,
                terminals,
                values,
                gamma=self._gamma,
                gae=self._gae,
                device=self.device,
            )

        # Mini-batch training
        batch_size = states.size(0)

        # List to track policy loss over minibatches
        losses = []
        actor_losses = []
        value_losses = []
        entropy_losses = []

        clip_fractions = []
        target_kl = []
        grad_dicts = []

        for k in range(self._K):
            indices = torch.randperm(batch_size)[: self.minibatch_size]
            mb_states, mb_actions = states[indices], actions[indices]
            mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

            # advantages
            mb_advantages = advantages[indices]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

            # 1. Critic Update (with optional regularization)
            mb_values = self.critic(mb_states)
            value_loss = self.mse_loss(mb_values, mb_returns)
            l2_reg = (
                sum(param.pow(2).sum() for param in self.critic.parameters())
                * self._l2_reg
            )
            value_loss += l2_reg

            # Track value loss for logging
            value_losses.append(value_loss.item())

            # 2. Policy Update
            _, metaData = self.policy(mb_states)
            logprobs = self.policy.log_prob(metaData["dist"], mb_actions)
            entropy = self.policy.entropy(metaData["dist"])
            ratios = torch.exp(logprobs - mb_old_logprobs)

            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = self._entropy_scaler * entropy.mean()

            # Track policy loss for logging
            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())

            # Total loss
            loss = actor_loss + 0.5 * value_loss - entropy_loss
            losses.append(loss.item())

            # Compute clip fraction (for logging)
            clip_fraction = torch.mean(
                (torch.abs(ratios - 1) > self._eps).float()
            ).item()
            clip_fractions.append(clip_fraction)

            # Check if KL divergence exceeds target KL for early stopping
            kl_div = torch.mean(mb_old_logprobs - logprobs)
            target_kl.append(kl_div.item())
            if kl_div.item() > self._target_kl:
                break

            # Update critic parameters
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            grad_dict = self.compute_gradient_norm(
                [self.policy, self.critic],
                ["policy", "critic"],
                dir="PPO",
                device=self.device,
            )
            grad_dicts.append(grad_dict)
            self.optimizer.step()

        # Logging
        loss_dict = {
            "PPO/loss": np.mean(losses),
            "PPO/actor_loss": np.mean(actor_losses),
            "PPO/value_loss": np.mean(value_losses),
            "PPO/entropy_loss": np.mean(entropy_losses),
            "PPO/clip_fraction": np.mean(clip_fractions),
            "PPO/klDivergence": target_kl[-1],
            "PPO/avg_reward": rewards.mean().item(),
            "PPO/K-epoch": k + 1,
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.policy, self.critic],
            ["policy", "critic"],
            dir="PPO",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        # Cleanup
        del states, actions, rewards, terminals, old_logprobs
        torch.cuda.empty_cache()

        self.eval()

        timesteps = self.minibatch_size * (k + 1)
        update_time = time.time() - t0
        return loss_dict, timesteps, update_time

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values and counts for each key
        sum_dict = {}
        count_dict = {}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                if key not in sum_dict:
                    sum_dict[key] = 0
                    count_dict[key] = 0
                sum_dict[key] += value
                count_dict[key] += 1

        # Calculate the average for each key
        avg_dict = {key: sum_val / count_dict[key] for key, sum_val in sum_dict.items()}

        return avg_dict

    def save_model(self, logdir, epoch=None, is_best=False):
        self.policy = self.policy.cpu()
        self.critic = self.critic.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (self.policy, self.critic),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.critic = self.critic.to(self.device)
