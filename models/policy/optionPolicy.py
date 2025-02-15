import time
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
from math import floor
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import fmin_l_bfgs_b as bfgs

from utils.torch import get_flat_grad_from, get_flat_params_from, set_flat_params_to
from utils.utils import estimate_advantages, estimate_psi
from models.layers.building_blocks import MLP
from models.layers.op_networks import OptionPolicy, OptionCritic
from models.policy.base_policy import BasePolicy


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


class OP_Controller(BasePolicy):
    def __init__(
        self,
        sf_network: BasePolicy,
        policy: OptionPolicy,
        critic: OptionCritic,
        sf_dim: int,
        snac_split_ratio: float,
        reward_options: np.ndarray,
        state_options: np.ndarray,
        alpha: int,
        minibatch_size: int,
        args,
    ):
        super(OP_Controller, self).__init__()

        # constants
        self._a_dim = args.a_dim
        self._entropy_scaler = args.op_entropy_scaler
        self._gamma = args.gamma
        self._l2_reg = args.weight_loss_scaler
        self._is_discrete = args.is_discrete
        self._minibatch_size = minibatch_size
        self._K = args.K_epochs
        self.mode = args.op_mode
        self.device = args.device

        # algorithmic commons
        self.sf_r_dim = floor(sf_dim * snac_split_ratio)
        self.sf_s_dim = sf_dim - self.sf_r_dim

        if reward_options is not None:
            self.reward_options = nn.Parameter(torch.tensor(reward_options)).to(
                dtype=self._dtype, device=self.device
            )
            self.num_reward_options = reward_options.shape[0]
        else:
            self.reward_options = None
            self.num_reward_options = 0

        if state_options is not None:
            self.state_options = nn.Parameter(torch.tensor(state_options)).to(
                dtype=self._dtype, device=self.device
            )
            self.num_state_options = state_options.shape[0]
        else:
            self.state_options = None
            self.num_state_options = 0

        self.num_subtasks = self.num_reward_options + self.num_state_options

        # params for training
        self.optimizers = {}

        if self.mode == "ppo":
            # PPO params
            self._target_kl = args.target_kl
            self._eps = args.eps_clip
            self._gae = args.gae
        elif self.mode == "sac":
            # SAC params
            self._tune_alpha = args.tune_alpha
            self._soft_update_rate = args.sac_soft_update_rate
            self._target_update_interval = args.target_update_interval
            self._target_entropy = -self._a_dim

            if self.mode == "sac":
                self.alpha = torch.full((self.num_subtasks,), alpha, device=self.device)
                if self._tune_alpha:
                    self.log_alpha = nn.Parameter(torch.log(self.alpha))
                    self.optimizers["alpha"] = torch.optim.Adam(
                        [self.log_alpha], lr=args.sac_alpha_lr
                    )

                self.target_critic = deepcopy(critic)

                self.num_update = 1
        else:
            raise NotImplementedError(
                f"{self.mode} mode is not implemented (ppo or sac)"
            )

        # trainable networks
        self.sf_network = sf_network
        self.policy = policy
        self.critic = critic

        self.optimizers["policy"] = torch.optim.Adam(
            self.policy.parameters(), lr=args.op_policy_lr
        )
        self.optimizers["critic"] = torch.optim.Adam(
            self.critic.parameters(), lr=args.op_critic_lr
        )

        # inherent variable
        self._forward_steps = 0

        #
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.sf_network.device = device
        self.to(device)

    def preprocess_obs(self, obs):
        observation = obs["observation"]

        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation).to(self._dtype).to(self.device)

        return {"observation": observation}

    def forward(self, obs, z, deterministic=False):
        """
        Image-based state dimension ~ [Batch, width, height, channel] or [width, height, channel]
        Flat tensor-based state dimension ~ [Batch, tensor] or [tensor]
        """
        self._forward_steps += 1
        obs = self.preprocess_obs(obs)

        a, metaData = self.policy(obs["observation"], z=z, deterministic=deterministic)

        with torch.no_grad():
            value, _ = self.critic(obs["observation"], z=z)
        option_term = True if value < 0 else False

        return a, {
            "option_termination": option_term,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self._soft_update_rate)
                + param.data * self._soft_update_rate
            )

    def learn(self, batch, z):
        if self.mode == "ppo":
            loss_dict, op_timesteps, update_time = self.ppo_learn(batch, z)
        elif self.mode == "sac":
            loss_dict, op_timesteps, update_time = self.sac_learn(batch, z)
        return loss_dict, op_timesteps, update_time

    def psuedo_reward(self, states, next_states, z):
        self.sf_network.eval()
        with torch.no_grad():
            phi = self.sf_network.get_features(states, deterministic=True)
            next_phi = self.sf_network.get_features(next_states, deterministic=True)
            deltaPhi = next_phi - phi

        is_reward_option = True if z < self.num_reward_options else False

        if is_reward_option:
            deltaPhi, _ = self.split(deltaPhi, self.sf_r_dim)
            weight = self.reward_options[z]
            # print(f"z: {z}, is_reward {is_reward_option}, {self.reward_options[z]}")
        else:
            z = z - self.num_reward_options
            _, deltaPhi = self.split(deltaPhi, self.sf_r_dim)
            weight = self.state_options[z]
            # print(f"z: {z}, is_reward {is_reward_option}, {self.state_options[z]}")

        pseudo_rewards = self.multiply_weights(deltaPhi, weight)

        pseudo_rewards *= 1e2  # (pseudo_rewards + 1e-3) / (1e-3 + 1e-3)
        return pseudo_rewards

    def sac_learn(self, batch, z):
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        next_states = to_tensor(batch["next_states"])
        terminals = to_tensor(batch["terminals"])

        # batch processing
        rewards = self.psuedo_reward(states, next_states, z)
        states = states.reshape(states.shape[0], -1)
        next_states = next_states.reshape(next_states.shape[0], -1)

        # Mini-batch training
        batch_size = states.size(0)

        actor_losses = []
        critic_losses = []
        alpha_loses = []

        actor_grad_dicts, critic_grad_dicts = [], []

        for _ in range(self._K):
            indices = torch.randperm(batch_size)[: self._minibatch_size]
            mb_states, mb_actions = states[indices], actions[indices]
            mb_next_states = next_states[indices]
            mb_rewards, mb_terminals = rewards[indices], terminals[indices]

            # Action by the current actor for the sampled state
            actions_pi, meta_pi = self.policy(mb_states, z)

            # Alpha Loss
            if self._tune_alpha:
                alpha_loss = -(
                    (
                        self.log_alpha[z]
                        * (meta_pi["logprobs"] + self._target_entropy).detach()
                    ).mean()
                )

                self.optimizers["alpha"].zero_grad()
                alpha_loss.backward()
                self.optimizers["alpha"].step()

                self.alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor(1e-5).to(self.device)

            # Track loss for logging
            alpha_loses.append(alpha_loss.item())

            # Critic Loss
            with torch.no_grad():
                next_actions, next_meta = self.policy(mb_next_states, z)
                next_q1, next_q2, _ = self.target_critic(
                    mb_next_states, next_actions, z
                )
                next_q = (
                    torch.min(next_q1, next_q2) - self.alpha[z] * next_meta["logprobs"]
                )
                target_q = mb_rewards + (1 - mb_terminals) * self._gamma * next_q

            q1, q2, _ = self.critic(mb_states, mb_actions, z)
            critic_loss = 0.5 * (
                self.mse_loss(q1, target_q) + self.mse_loss(q2, target_q)
            )

            # Track loss for logging
            critic_losses.append(critic_loss.item())

            self.optimizers["critic"].zero_grad()
            critic_loss.backward()
            critic_grad_dict = self.compute_gradient_norm(
                [self.critic],
                ["critic"],
                dir="OP_SAC",
                device=self.device,
            )
            critic_grad_dict.append(critic_grad_dict)
            self.optimizers["critic"].step()

            q1_pi, q2_pi, _ = self.critic(mb_states, actions_pi, z)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha[z] * meta_pi["logprobs"] - q_pi).mean()

            # Track loss for logging
            actor_losses.append(actor_loss.item())

            self.optimizers["policy"].zero_grad()
            actor_loss.backward()
            actor_grad_dict = self.compute_gradient_norm(
                [self.policy],
                ["policy"],
                dir="OP_SAC",
                device=self.device,
            )
            actor_grad_dicts.append(actor_grad_dict)
            self.optimizers["policy"].step()

            # Soft update of target networks
            if self.num_update % self._target_update_interval == 0:
                self.soft_update(self.target_critic, self.critic)

            self.num_update += 1

        # Log losses
        loss_dict = {
            "OP_SAC/critic_loss": np.mean(critic_losses),
            "OP_SAC/policy_loss": np.mean(actor_losses),
            "OP_SAC/alpha_loss": np.mean(alpha_loses),
            f"OP_SAC/alpha {z}": self.alpha[z].item(),
            f"OP_SAC/avg_reward{z}": rewards.mean().item(),
        }
        actor_grad_dict = self.average_dict_values(actor_grad_dicts)
        critic_grad_dict = self.average_dict_values(critic_grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.policy, self.critic],
            ["policy", "critic"],
            dir="OP_SAC",
            device=self.device,
        )
        loss_dict.update(norm_dict)
        loss_dict.update(actor_grad_dict)
        loss_dict.update(critic_grad_dict)

        update_time = time.time() - t0
        return loss_dict, update_time

    def ppo_learn(self, batch, z):
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"])
        actions = to_tensor(batch["actions"])
        next_states = to_tensor(batch["next_states"])
        terminals = to_tensor(batch["terminals"])
        old_logprobs = to_tensor(batch["logprobs"])

        # batch processing
        rewards = self.psuedo_reward(states, next_states, z)
        states = states.reshape(states.shape[0], -1)
        next_states = next_states.reshape(next_states.shape[0], -1)

        # Compute advantages and returns
        with torch.no_grad():
            values, _ = self.critic(states, z)
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

        # K - Loop
        for k in range(self._K):
            indices = torch.randperm(batch_size)[: self._minibatch_size]
            mb_states, mb_actions = states[indices], actions[indices]
            mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

            # advantages
            mb_advantages = advantages[indices]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

            # 1. Critic Update (with optional regularization)
            mb_values, _ = self.critic(mb_states, z)
            value_loss = self.mse_loss(mb_values, mb_returns)
            l2_reg = sum(param.pow(2).sum() for param in self.critic.parameters())
            value_loss += self._l2_reg * l2_reg

            # Track value loss for logging
            value_losses.append(value_loss.item())

            # 2. Policy Update
            _, metaData = self.policy(mb_states, z)
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

            for _, optim in self.optimizers.items():
                optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            grad_dict = self.compute_gradient_norm(
                [self.policy, self.critic],
                ["optionPolicy", "optionCritic"],
                dir="OP_PPO",
                device=self.device,
            )
            grad_dicts.append(grad_dict)
            for _, optim in self.optimizers.items():
                optim.step()

        norm_dict = self.compute_weight_norm(
            [self.policy, self.critic],
            ["policy", "critic"],
            dir="OP_PPO",
            device=self.device,
        )

        # Logging
        loss_dict = {
            "OP_PPO/loss/total_loss": np.mean(losses),
            "OP_PPO/loss/actor_loss": np.mean(actor_losses),
            "OP_PPO/loss/value_loss": np.mean(value_losses),
            "OP_PPO/loss/entropy_loss": np.mean(entropy_losses),
            "OP_PPO/analytics/clip_fraction": np.mean(clip_fractions),
            "OP_PPO/analytics/klDivergence": target_kl[-1],
            "OP_PPO/analytics/K-epoch": k + 1,
            f"OP_PPO/measure/avg_rewards{z}": (
                torch.sum(rewards) / torch.sum(terminals)
            ).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.policy, self.critic],
            ["policy", "critic"],
            dir="OP_PPO",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        del (
            states,
            actions,
            rewards,
            terminals,
            old_logprobs,
        )
        torch.cuda.empty_cache()

        self.eval()

        timesteps = self._minibatch_size * (k + 1)
        update_time = time.time() - t0
        return (loss_dict, timesteps, update_time)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        # Initialize a dictionary to hold the sum of values for each key
        sum_dict = {key: 0 for key in dict_list[0].keys()}

        # Iterate over each dictionary in the list
        for d in dict_list:
            for key, value in d.items():
                sum_dict[key] += value

        # Calculate the average for each key
        avg_dict = {key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()}

        return avg_dict

    def save_model(self, logdir, epoch=None, is_best=False):
        self.policy = self.policy.cpu()
        self.critic = self.critic.cpu()
        reward_options = (
            self.reward_options.detach().clone().cpu().numpy()
            if self.reward_options is not None
            else self.reward_options
        )
        state_options = (
            self.state_options.detach().clone().cpu().numpy()
            if self.state_options is not None
            else self.state_options
        )

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "bestmodel.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")

        if self.mode == "sac":
            alpha = self.alpha.clone().cpu()
            pickle.dump(
                (
                    self.policy,
                    self.critic,
                    reward_options,
                    state_options,
                    alpha,
                ),
                open(path, "wb"),
            )
        elif self.mode == "ppo":
            pickle.dump(
                (
                    self.policy,
                    self.critic,
                    reward_options,
                    state_options,
                ),
                open(path, "wb"),
            )

        self.policy = self.policy.to(self.device)
        self.critic = self.critic.to(self.device)
