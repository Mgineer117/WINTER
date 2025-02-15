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
from utils.utils import estimate_advantages
from models.layers.building_blocks import MLP
from models.policy.optionPolicy import OP_Controller
from models.layers.hc_networks import HC_Policy, HC_PPO, HC_Critic
from models.policy.base_policy import BasePolicy


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


def compare_network_weights(model1: nn.Module, model2: nn.Module) -> float:
    """
    Compare the weights of two models and return the mean squared error between them.

    Args:
        model1 (nn.Module): The first model to compare.
        model2 (nn.Module): The second model to compare.

    Returns:
        float: The mean squared error between the weights of the two models.
    """
    mse_loss = nn.MSELoss()
    total_mse = 0.0
    num_params = 0

    # Iterate through parameters of both models
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if param1.shape != param2.shape:
            raise ValueError(
                "Model parameters have different shapes, models might have different architectures."
            )

        # Calculate MSE between parameters
        mse = mse_loss(param1, param2)
        total_mse += mse.item()
        num_params += 1

    # Average MSE across all parameters
    average_mse = total_mse / num_params if num_params > 0 else 0.0
    print(average_mse)

    return average_mse


class HC_Controller(BasePolicy):
    def __init__(
        self,
        sf_network: BasePolicy,
        op_network: OP_Controller,
        policy: HC_Policy,
        primitivePolicy: HC_PPO,
        critic: HC_Critic,
        minibatch_size: int,
        a_dim: int,
        policy_lr: float = 5e-4,
        critic_lr: float = 1e-4,
        entropy_scaler: float = 1e-3,
        eps: float = 0.2,
        gae: float = 0.95,
        gamma: float = 0.99,
        K: int = 5,
        target_kl: float = 0.03,
        l2_reg: float = 1e-6,
        device: str = "cpu",
    ):
        super(HC_Controller, self).__init__()
        # constants
        self.device = device
        self.minibatch_size = minibatch_size

        self._num_weights = policy._num_weights
        self._entropy_scaler = entropy_scaler
        self._a_dim = a_dim
        self._eps = eps
        self._gamma = gamma
        self._gae = gae
        self._K = K
        self._l2_reg = l2_reg
        self._target_kl = target_kl
        self._forward_steps = 0

        # trainable networks
        self.policy = policy
        self.primitivePolicy = primitivePolicy
        self.critic = critic

        self.sf_network = sf_network
        self.op_network = op_network

        self.optimizers = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": policy_lr},
                {"params": self.primitivePolicy.parameters(), "lr": policy_lr},
                {"params": self.critic.parameters(), "lr": critic_lr},
            ]
        )

        #
        self.dummy = torch.tensor(1e-10)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.sf_network.device = device
        self.op_network.device = device
        self.to(device)

    def getPhi(self, obs):
        obs = {"observation": obs}
        with torch.no_grad():
            phi = self.sf_network.get_features(obs, deterministic=True)
        return phi

    def preprocess_obs(self, obs):
        observation = obs["observation"]

        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation).to(self._dtype).to(self.device)

        return {"observation": observation}

    def forward(self, obs, idx=None, deterministic=False):
        """
        Image-based state dimension ~ [Batch, width, height, channel] or [width, height, channel]
        Flat tensor-based state dimension ~ [Batch, tensor] or [tensor]
        """
        self._forward_steps += 1

        obs = self.preprocess_obs(obs)

        if idx is None:
            # sample a from the Hierarchical Policy
            z, z_argmax, metaData = self.policy(
                obs["observation"], deterministic=deterministic
            )
        else:
            # keep using the given z
            z = F.one_hot(idx, num_classes=self.policy._a_dim)
            z_argmax = idx
            probs = torch.tensor(1.0)
            metaData = {
                "probs": probs,
                "logprobs": torch.log(probs),
                "entropy": self.dummy,
            }  # dummy

        # print(z_argmax, self._num_weights)
        is_option = True if z_argmax < self._num_weights else False

        if is_option:
            # option selection
            with torch.no_grad():
                a, option_dict = self.op_network(obs, z_argmax, deterministic=True)
            option_termination = option_dict["option_termination"]
        else:
            # primitive action selection
            a, _ = self.primitivePolicy(obs["observation"], deterministic=deterministic)
            option_termination = True

        return a, {
            "z": z,
            "z_argmax": z_argmax,
            "is_option": is_option,
            "is_hc_controller": True,
            "option_termination": option_termination,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
        }

    def learn(self, batch):
        self.train()
        t0 = time.time()

        # Ingredients: Convert batch data to tensors
        def to_tensor(data):
            return torch.from_numpy(data).to(self._dtype).to(self.device)

        states = to_tensor(batch["states"]).reshape(batch["states"].shape[0], -1)
        actions = to_tensor(batch["actions"])
        option_actions = to_tensor(batch["option_actions"])
        rewards = to_tensor(batch["rewards"])
        terminals = to_tensor(batch["terminals"])
        old_logprobs = to_tensor(batch["logprobs"])

        # pm ingredients
        if isinstance(self.primitivePolicy, HC_PPO):
            with torch.no_grad():
                actions, pm_metaData = self.primitivePolicy(states)
                pm_old_logprobs = self.primitivePolicy.log_prob(
                    pm_metaData["dist"], actions
                )

        # Minibatch setup
        batch_size = states.size(0)

        # Compute advantages and returns
        with torch.no_grad():
            values, _ = self.critic(states)
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
        losses, actor_losses = [], []
        value_losses, entropy_losses = [], []

        clip_fractions, target_kl, grad_dicts = [], [], []

        # K - Loop
        for k in range(self._K):
            indices = torch.randperm(batch_size)[: self.minibatch_size]
            mb_states = states[indices]
            mb_option_actions = option_actions[indices]
            mb_old_logprobs, mb_returns = old_logprobs[indices], returns[indices]

            # global batch normalization and target return
            mb_advantages = advantages[indices]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

            # 1. Critic Update (with optional regularization)
            mb_values, _ = self.critic(mb_states)
            valueLoss = self.mse_loss(mb_returns, mb_values)
            value_losses.append(valueLoss.item())

            # 2. Policy Update
            _, _, metaData = self.policy(mb_states)
            # Compute hierarchical policy logprobs and entropy
            logprobs = self.policy.log_prob(metaData["dist"], mb_option_actions)
            entropy = self.policy.entropy(metaData["dist"])
            ratios = torch.exp(logprobs - mb_old_logprobs)

            # prepare updates
            surr1 = ratios * mb_advantages
            surr2 = torch.clamp(ratios, 1 - self._eps, 1 + self._eps) * mb_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = self._entropy_scaler * entropy.mean()

            if isinstance(self.primitivePolicy, HC_PPO):
                # find mask: the actions contributions by hc policy only
                pm_mask = torch.argmax(mb_option_actions, dim=-1) == self._num_weights

                if pm_mask is not None:
                    mb_actions = actions[indices]
                    mb_pm_old_logprobs = pm_old_logprobs[indices]

                    _, pm_metaData = self.primitivePolicy(mb_states[pm_mask])

                    pm_logprobs = self.policy.log_prob(
                        pm_metaData["dist"], mb_actions[pm_mask]
                    )
                    pm_entropy = self.policy.entropy(pm_metaData["dist"])
                    pm_ratios = torch.exp(pm_logprobs - mb_pm_old_logprobs[pm_mask])

                    # prepare updates
                    surr1 = pm_ratios * mb_advantages[pm_mask]
                    surr2 = (
                        torch.clamp(pm_ratios, 1 - self._eps, 1 + self._eps)
                        * mb_advantages[pm_mask]
                    )

                    pm_actorLoss = -torch.min(surr1, surr2).mean()
                    pm_entropyLoss = self._entropy_scaler * pm_entropy.mean()

                    # finishing up
                    actor_loss += pm_actorLoss
                    entropy_loss += pm_entropyLoss

                    # Simplify the repeated patterns
                    ratios = torch.cat((ratios, pm_ratios), dim=0)
                    logprobs = torch.cat((logprobs, pm_logprobs), dim=0)
                    mb_old_logprobs = torch.cat(
                        (mb_old_logprobs, mb_pm_old_logprobs[pm_mask]), dim=0
                    )

            loss = actor_loss + 0.5 * valueLoss - entropy_loss

            losses.append(loss.item())
            actor_losses.append(actor_loss.item())
            entropy_losses.append(entropy_loss.item())

            # Compute clip fraction (for logging)
            clip_fraction = torch.mean((torch.abs(ratios - 1) > self._eps).float())
            clip_fractions.append(clip_fraction.item())

            # Check if KL divergence exceeds target KL for early stopping
            kl_div = torch.mean(mb_old_logprobs - logprobs)
            target_kl.append(kl_div.item())
            if kl_div.item() > self._target_kl:
                break

            self.optimizers.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            grad_dict = self.compute_gradient_norm(
                [self.policy, self.primitivePolicy, self.critic],
                ["policy", "primitivePolicy", "critic"],
                dir="HC",
                device=self.device,
            )
            self.optimizers.step()

        # Logging
        loss_dict = {
            "HC/loss/loss": np.mean(losses),
            "HC/loss/actor_loss": np.mean(actor_losses),
            "HC/loss/value_loss": np.mean(value_losses),
            "HC/loss/entropy_loss": np.mean(entropy_losses),
            "HC/analytics/clip_fraction": np.mean(clip_fractions),
            "HC/analytics/klDivergence": target_kl[-1],
            "HC/analytics/K-epoch": k + 1,
            "HC/measure/avg_rewards": (
                torch.sum(rewards) / torch.sum(terminals)
            ).item(),
        }
        grad_dict = self.average_dict_values(grad_dicts)
        norm_dict = self.compute_weight_norm(
            [self.policy, self.primitivePolicy, self.critic],
            ["policy", "primitivePolicy", "critic"],
            dir="HC",
            device=self.device,
        )
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        del states, actions, option_actions, rewards, terminals, old_logprobs
        torch.cuda.empty_cache()

        self.eval()

        timesteps = self.minibatch_size * (k + 1)
        update_time = time.time() - t0

        return loss_dict, timesteps, update_time

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

    def save_model(
        self, logdir: str, epoch: int = None, name: str = None, is_best=False
    ):
        self.policy = self.policy.cpu()
        self.primitivePolicy = self.primitivePolicy.cpu()
        self.critic = self.critic.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, f"best_model_{name}.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (self.policy, self.primitivePolicy, self.critic),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.primitivePolicy = self.primitivePolicy.to(self.device)
        self.critic = self.critic.to(self.device)
