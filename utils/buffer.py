import numpy as np
import torch

from math import ceil, floor
from typing import Optional, Union, Tuple, Dict, List


class TrajectoryBuffer:
    def __init__(
        self,
        state_dim: tuple,
        action_dim: int,
        hc_action_dim: int,
        episode_len: int,
        min_batch_size: int,
        max_batch_size: int,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hc_action_dim = hc_action_dim
        self.episode_len = episode_len
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Using lists to store trajectories
        self.samples = {
            "states": np.full(
                (self.max_batch_size,) + self.state_dim, np.nan, dtype=np.float32
            ),
            "actions": np.full(
                (self.max_batch_size, self.action_dim), np.nan, dtype=np.float32
            ),
            "option_actions": np.full(
                (self.max_batch_size, self.hc_action_dim), np.nan, dtype=np.float32
            ),
            "next_states": np.full(
                (self.max_batch_size,) + self.state_dim, np.nan, dtype=np.float32
            ),
            "rewards": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
            "terminals": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
            "logprobs": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
            "entropys": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
        }
        self.num_samples = 0
        self.idx = 0
        self.full = False

    def update_samples(self, batch: dict):
        batch_size = batch["rewards"].shape[0]
        batch_deficiency = self.max_batch_size - self.idx

        if batch_deficiency < batch_size:
            end_idx = self.max_batch_size
            for k in batch.keys():
                self.samples[k][self.idx : end_idx] = batch[k][:batch_deficiency]
            self.idx = 0

            # remaining data
            end_idx = self.idx + (batch_size - batch_deficiency)
            for k in batch.keys():
                self.samples[k][self.idx : end_idx] = batch[k][batch_deficiency:]
            self.idx = end_idx

            self.num_samples = self.max_batch_size
            self.full = True
        else:
            end_idx = self.idx + batch_size
            for k in batch.keys():
                self.samples[k][self.idx : end_idx] = batch[k]
            self.idx = end_idx

            self.num_samples += batch_size

    def push(self, batch: dict, post_process: str | None = None) -> None:
        """
        Method: Push the batch into the data buffer. This saves it as a trajectory
        --------------------------------------------------------------------------------------------
        Input: batch --> dict-type with key of states, actions, next_states, rewards, masks
                        // mask = not done in gym context
        Output: None
        """
        if post_process == "nonzero_rewards":
            nonzero_indices = np.nonzero(batch["rewards"])[0]
            for k in self.samples.keys():
                batch[k] = batch[k][nonzero_indices]

        self.update_samples(batch)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        sample the data from the buffer
        """
        if batch_size > self.num_samples:
            raise ValueError(
                f"The given size {batch_size} exceeds the buffer {self.num_samples}"
            )

        sampled_batch = {key: None for key in self.samples.keys()}
        sample_indices = np.arange(self.num_samples)
        chosen_indices = np.random.choice(
            sample_indices, size=batch_size, replace=False
        )

        for k in sampled_batch.keys():
            sampled_batch[k] = self.samples[k][chosen_indices]

        return sampled_batch

    def wipe(self):
        self.samples = {
            "states": np.full(
                (self.max_batch_size,) + self.state_dim, np.nan, dtype=np.float32
            ),
            "actions": np.full(
                (self.max_batch_size, self.action_dim), np.nan, dtype=np.float32
            ),
            "option_actions": np.full(
                (self.max_batch_size, self.hc_action_dim), np.nan, dtype=np.float32
            ),
            "next_states": np.full(
                (self.max_batch_size,) + self.state_dim, np.nan, dtype=np.float32
            ),
            "rewards": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
            "terminals": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
            "logprobs": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
            "entropys": np.full((self.max_batch_size, 1), np.nan, dtype=np.float32),
        }
        self.num_samples = 0
        self.idx = 0
        self.full = False

    def sample_all(self) -> Dict[str, torch.Tensor]:
        return self.samples
