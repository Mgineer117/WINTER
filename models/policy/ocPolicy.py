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
from models.layers.sf_networks import ConvNetwork
from models.layers.oc_networks import OC_Policy, OC_Critic
from models.policy.base_policy import BasePolicy


class OC_Learner(BasePolicy):
    def __init__(
        self,
        policy: OC_Policy,
        critic: OC_Critic,
        policy_lr: float = 3e-4,
        critic_lr: float = 5e-4,
        entropy_scaler: float = 1e-3,
        termination_reg: float = 1e-2,
        gamma: float = 0.99,
        K: int = 5,
        device: str = "cpu",
    ):
        super(OC_Learner, self).__init__()

        # constants
        self.device = device

        self._a_dim = policy._a_dim
        self._entropy_scaler = entropy_scaler
        self._termination_reg = termination_reg
        self._gamma = gamma
        self._l2_reg = 1e-6
        self._bfgs_iter = K
        self._forward_steps = 0

        # trainable networks
        self.policy = policy
        self.critic = critic
        self.target_critic = deepcopy(self.critic)

        # Option-critic's critic is not mse-based
        # so BFGS is highly unstable -> ADAM
        policy_lr = 2e-4
        critic_lr = 2e-4

        if critic_lr is None:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
            self.is_bfgs = True
        else:
            self.optimizer = torch.optim.Adam(
                [
                    {"params": self.policy.parameters(), "lr": policy_lr},
                    {"params": self.critic.parameters(), "lr": critic_lr},
                ]
            )
            self.is_bfgs = False

        #
        self.dummy = torch.tensor(0.0)
        self.to(self.device)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def preprocess_obs(self, obs):
        observation = obs["observation"]
        agent_pos = obs["agent_pos"]

        # preprocessing
        observation = torch.from_numpy(observation).to(self._dtype).to(self.device)

        if np.any(agent_pos != None):
            agent_pos = torch.from_numpy(agent_pos).to(self._dtype).to(self.device)

        return {"observation": observation, "agent_pos": agent_pos}

    def predict_option_termination(self, next_obs: torch.Tensor, z: int):
        next_obs = self.preprocess_obs(next_obs)
        option_termination = self.policy.predict_option_termination(
            next_obs["observation"], z=z
        )
        return option_termination

    def forward(self, obs, z=None, deterministic=False):
        """
        Image-based state dimension ~ [Batch, width, height, channel] or [width, height, channel]
        Flat tensor-based state dimension ~ [Batch, tensor] or [tensor]
        z is dummy input for code consistency
        """
        self._forward_steps += 1
        obs = self.preprocess_obs(obs)

        # deterministic is eval signal in this code
        epsilon = self.policy.epsilon(is_eval=deterministic)

        # the first iteration where z is not given
        if z is None:
            greedy_option = self.critic.greedy_option(obs["observation"])
        else:
            greedy_option = z

        current_option = (
            torch.randint(self.policy._num_options, (1,))
            if np.random.rand() < epsilon
            else greedy_option
        ).squeeze()
        z = F.one_hot(current_option, num_classes=self.policy._num_options)

        a, metaData = self.policy(
            obs["observation"], z=current_option, deterministic=deterministic
        )

        return a, {
            "z": z,
            "z_argmax": current_option,
            "probs": metaData["probs"],
            "logprobs": metaData["logprobs"],
            "entropy": metaData["entropy"],
            "is_option": True,
            "is_hc_controller": False,
        }

    def actor_loss(self, batch):
        """Train policy and terminations

        Args:
            states (_type_): _description_
        """
        states = self.to_tensor(batch["states"], self._dtype, self.device)
        next_states = self.to_tensor(batch["next_states"], self._dtype, self.device)
        option_actions = self.to_tensor(
            batch["option_actions"], self._dtype, self.device
        )
        rewards = self.to_tensor(batch["rewards"], self._dtype, self.device)
        terminals = self.to_tensor(batch["terminals"], self._dtype, self.device)

        logprobs = torch.empty(rewards.shape).to(self.device)
        entropys = torch.empty(rewards.shape).to(self.device)

        for i in range(states.shape[0]):
            state = states[i]
            option_action = torch.argmax(option_actions[i], dim=-1).long()

            a, metaData = self.policy(state=state, z=option_action)

            logprobs[i] = self.policy.log_prob(dist=metaData["dist"], actions=a)
            entropys[i] = self.policy.entropy(dist=metaData["dist"])

        option_term_prob = self.policy.get_terminations(states)
        option_term_prob = self.multiply_options(option_term_prob, option_actions)

        next_option_term_prob = self.policy.get_terminations(next_states).detach()
        next_option_term_prob = self.multiply_options(
            next_option_term_prob, option_actions
        )

        Q = self.critic(states).detach()
        next_Q_prime = self.target_critic(next_states).detach()

        Q_by_option = self.multiply_options(Q, option_actions)
        Q_by_max = torch.max(Q, dim=-1, keepdim=True)[0]

        next_Q_prime_by_option = self.multiply_options(next_Q_prime, option_actions)
        next_Q_prime_by_max = torch.max(next_Q_prime, dim=-1, keepdim=True)[0]

        # Target update gt
        gt = rewards + (1 - terminals) * self._gamma * (
            (1 - next_option_term_prob) * next_Q_prime_by_option
            + next_option_term_prob * next_Q_prime_by_max
        )

        # The termination loss
        termination_loss = (
            option_term_prob
            * (Q_by_option.detach() - Q_by_max.detach() + self._termination_reg)
            * (1 - terminals)
        )
        # actor-critic policy gradient with entropy regularization
        entropy_loss = self._entropy_scaler * entropys
        policy_loss = -logprobs * (gt.detach() - Q_by_option.detach())

        actor_loss = torch.mean(termination_loss + policy_loss - entropy_loss)

        del states, next_states, option_actions, rewards, terminals, logprobs, entropys
        torch.cuda.empty_cache()

        return actor_loss, {
            "termination_loss": termination_loss,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
        }

    def critic_loss(self, batch):
        """
        Training Q
        """
        # normalization
        if self.normalizer is not None:
            batch["states"] = self.normalizer.normalize(batch["states"], update=False)
            batch["next_states"] = self.normalizer.normalize(
                batch["next_states"], update=False
            )

        states = self.to_tensor(batch["states"], self._dtype, self.device)
        next_states = self.to_tensor(batch["next_states"], self._dtype, self.device)
        option_actions = self.to_tensor(
            batch["option_actions"], self._dtype, self.device
        )
        rewards = self.to_tensor(batch["rewards"], self._dtype, self.device)
        terminals = self.to_tensor(batch["terminals"], self._dtype, self.device)

        Q = self.critic(states)
        next_Q_prime = self.target_critic(next_states).detach()

        Q_by_option = self.multiply_options(Q, option_actions)

        next_Q_prime_by_option = self.multiply_options(next_Q_prime, option_actions)
        next_Q_prime_by_max = torch.max(next_Q_prime, dim=-1, keepdim=True)[0]

        next_option_term_prob = self.policy.get_terminations(next_states).detach()
        next_option_term_prob = self.multiply_options(
            next_option_term_prob, option_actions
        )

        # Target update gt
        gt = rewards + (1 - terminals) * self._gamma * (
            (1 - next_option_term_prob) * next_Q_prime_by_option
            + next_option_term_prob * next_Q_prime_by_max
        )

        # to update Q we want to use the actual network, not the prime
        td_err = (Q_by_option - gt.detach()).pow(2).mul(0.5).mean()

        del states, next_states, option_actions, rewards, terminals
        torch.cuda.empty_cache()

        return td_err, {}

    def learn_policy(self, batch):
        """_summary_

        Args:
            batch (_type_): Online batch from sampler
        """
        t0 = time.time()

        self.optimizer.zero_grad()
        actorLoss, metaData = self.actor_loss(batch)
        actorLoss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        grad_dict = self.compute_gradient_norm(
            [self.policy],
            ["policy"],
            dir="OC",
            device=self.device,
        )
        norm_dict = self.compute_weight_norm(
            [self.policy],
            ["policy"],
            dir="OC",
            device=self.device,
        )
        self.optimizer.step()

        loss_dict = {
            "OC/actorLoss": actorLoss.item(),
            "OC/terminationLoss": torch.mean(metaData["termination_loss"]).item(),
            "OC/policyLoss": torch.mean(metaData["policy_loss"]).item(),
            "OC/entropyLoss": torch.mean(metaData["entropy_loss"]).item(),
            "OC/trainAvgReward": np.sum(batch["rewards"]) / np.sum(batch["terminals"]),
        }
        loss_dict.update(grad_dict)
        loss_dict.update(norm_dict)

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def learn_critic(self, batch, merge_critic: bool = False):
        """_summary_

        Args:
            batch (_type_): offline batch but live
        """
        t0 = time.time()

        if self.is_bfgs:
            # L-BFGS-F value network update
            def closure(flat_params):
                set_flat_params_to(self.policy, torch.tensor(flat_params))
                self.critic.zero_grad()
                value_loss, _ = self.critic_loss(batch)
                value_loss += sum(
                    param.pow(2).sum() * self._l2_reg
                    for param in self.critic.parameters()
                )
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

                return (
                    value_loss.item(),
                    get_flat_grad_from(self.critic.parameters()).cpu().numpy(),
                )

            flat_params, _, opt_info = bfgs(
                closure,
                get_flat_params_from(self.critic).detach().cpu().numpy(),
                maxiter=self._bfgs_iter,
            )
            set_flat_params_to(self.critic, torch.tensor(flat_params))
        else:
            # Divide batch into sub-batches for non-BFGS optimization
            sub_batches = self.divide_into_subbatches(batch, self._bfgs_iter)
            for minibatch in sub_batches:
                # Convert sub-batch data to tensors

                # Update critic with minibatch
                self.optimizer.zero_grad()
                value_loss, _ = self.critic_loss(minibatch)
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.optimizer.step()

        # migrate the parameters
        if merge_critic:
            self.target_critic.load_state_dict(self.critic.state_dict())

        # Compute advantages and returns
        with torch.no_grad():
            valueLoss, _ = self.critic_loss(batch)

        norm_dict = self.compute_weight_norm(
            [self.critic],
            ["critic"],
            dir="OC",
            device=self.device,
        )

        loss_dict = {
            "OC/valueLoss": valueLoss.item(),
        }
        loss_dict.update(norm_dict)

        t1 = time.time()
        self.eval()
        return (
            loss_dict,
            t1 - t0,
        )

    def divide_into_subbatches(self, batch, subbatch_size):
        """
        Divide a batch of dictionaries into sub-batches.

        Args:
            batch (dict): A dictionary where each value is a list or tensor of equal length.
            subbatch_size (int): The size of each sub-batch.

        Returns:
            List[dict]: A list of dictionaries representing sub-batches.
        """
        keys = batch.keys()
        num_samples = len(next(iter(batch.values())))  # Get the size of the batch
        subbatches = []

        for i in range(0, num_samples, subbatch_size):
            subbatch = {
                key: value[i : i + subbatch_size] for key, value in batch.items()
            }
            subbatches.append(subbatch)

        return subbatches

    def to_tensor(self, data, dtype, device):
        return torch.from_numpy(data).to(dtype).to(device)

    def save_model(self, logdir, epoch=None, is_best=False):
        self.policy = self.policy.cpu()
        self.critic = self.critic.cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (self.policy, self.critic, self.normalizer),
            open(path, "wb"),
        )
        self.policy = self.policy.to(self.device)
        self.critic = self.critic.to(self.device)
