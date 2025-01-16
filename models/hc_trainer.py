import time
import math
import os
import wandb
import pickle
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from typing import Optional, Dict, List
from tqdm.auto import trange
from collections import deque
from log.wandb_logger import WandbLogger
from models.policy.base_policy import BasePolicy
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# model-free policy trainer
class HCTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        prefix: str = "HC",
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 200,
        eval_episodes: int = 10,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        log_interval: int = 2,
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        # training parameters
        self._prefix = prefix
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._init_epoch = init_epoch
        self._eval_episodes = eval_episodes
        self.lr_scheduler = lr_scheduler

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.grid_type = grid_type

    def train(self) -> Dict[str, float]:
        start_time = time.time()
        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # Total iterations tracking
        total_iterations = (self._epoch - self._init_epoch) * self._step_per_epoch
        completed_iterations = 0

        # Train loop
        for e in trange(self._init_epoch, self._epoch, desc=f"HC Epoch"):
            ### Training loop
            self.policy.train()
            for it in trange(self._step_per_epoch, desc=f"Training", leave=False):
                batch, sample_time = self.sampler.collect_samples(
                    self.policy, grid_type=self.grid_type, is_option=True
                )
                loss, update_time = self.policy.learn(batch, prefix=self._prefix)

                # Calculate expected remaining time
                completed_iterations += 1
                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / completed_iterations
                remaining_time = avg_time_per_iter * (
                    total_iterations - completed_iterations
                )

                # Update environment steps and calculate time metrics
                self.num_env_steps += len(batch["rewards"])
                loss[self._prefix + "/sample_time"] = sample_time
                loss[self._prefix + "/update_time"] = update_time
                loss[self._prefix + "/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss, iter_idx=int(e * self._step_per_epoch + it))
                torch.cuda.empty_cache()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            ### Eval Loop
            self.policy.eval()
            eval_dict = self.evaluator(
                self.policy,
                env_step=self.num_env_steps,
                epoch=e + 1,
                iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                dir_name=self._prefix,
                grid_type=self.grid_type,
            )

            self.last_reward_mean.append(eval_dict["rew_mean"])
            self.last_reward_std.append(eval_dict["rew_std"])

            self.save_model(e + 1)
            torch.cuda.empty_cache()

        self.logger.print(
            f"total {self._prefix} training time: {((time.time() - start_time) / 3600):.2f} hours"
        )

    def save_model(self, e):
        # save checkpoint
        if e % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dirs[2], e)

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            self.policy.save_model(
                self.logger.log_dirs[2], e, name=self._prefix, is_best=True
            )
            self.last_max_reward = np.mean(self.last_reward_mean)

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

    def write_log(self, logging_dict: dict, iter_idx: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(int(iter_idx), display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, int(iter_idx))
