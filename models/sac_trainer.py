import time
import random
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
from models.policy.sacPolicy import SAC_Learner
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# Soft Actor-Critic (SAC) trainer
class SACTrainer:
    def __init__(
        self,
        policy: SAC_Learner,
        sampler: OnlineSampler,
        buffer: TrajectoryBuffer,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        batch_size: int = 256,
        eval_episodes: int = 10,
        log_interval: int = 2,
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        # Training parameters
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._init_epoch = init_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes

        # Initialize training components
        self.num_env_steps = 0
        self.log_interval = log_interval
        self.grid_type = grid_type
        self.last_max_reward = -1e10
        self.std_limit = 0.5

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        total_iterations = (self._epoch - self._init_epoch) * self._step_per_epoch
        completed_iterations = 0

        sample_time = self.warm_buffer()
        for e in trange(self._init_epoch, self._epoch, desc="SAC Epoch"):
            ### Training Loop
            self.policy.train()

            for it in trange(self._step_per_epoch, desc="Training", leave=False):
                batch = self.buffer.sample(self.policy.batch_size)
                loss_dict, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                completed_iterations += 1
                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / completed_iterations
                remaining_time = avg_time_per_iter * (
                    total_iterations - completed_iterations
                )

                # Update environment steps and calculate metrics
                loss_dict["SAC/sample_time"] = sample_time
                loss_dict["SAC/update_time"] = update_time
                loss_dict["SAC/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(
                    loss_dict,
                    iter_idx=int(e * self._step_per_epoch + it),
                )

                completed_iterations += 1
                sample_time = 0
                torch.cuda.empty_cache()

            # update the buffer
            batch, sample_time = self.sampler.collect_samples(
                self.policy,
                grid_type=self.grid_type,
            )
            self.buffer.push(batch)
            self.num_env_steps += len(batch["rewards"])

            if e % self.log_interval == 0:
                ### Evaluation Loop
                self.policy.eval()
                eval_dict = self.evaluator(
                    self.policy,
                    env_step=self.num_env_steps,
                    epoch=e + 1,
                    iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                    dir_name="SAC",
                    grid_type=self.grid_type,
                )

                self.last_reward_mean = eval_dict["rew_mean"]
                self.save_model(e + 1)

        total_time = (time.time() - start_time) / 3600
        self.logger.print(f"Total SAC training time: {total_time:.2f} hours")

        return self._epoch

    def warm_buffer(self, verbose=True):
        t0 = time.time()
        # make sure there is nothing there
        self.buffer.wipe()

        # collect enough batch
        count = 0
        total_sample_time = 0
        sample_time = 0
        while self.buffer.num_trj() < self.buffer.min_num_trj:
            batch, sampleT = self.sampler.collect_samples(
                self.policy, grid_type=self.grid_type
            )
            self.buffer.push(batch)
            self.num_env_steps += len(batch["rewards"])
            sample_time += sampleT
            total_sample_time += sampleT
            if count % 50 == 0:
                if verbose:
                    print(
                        f"\nWarming buffer {self.buffer.num_trj()}/{self.buffer.min_num_trj} | sample_time = {sample_time:.2f}s",
                        end="",
                    )
                sample_time = 0
            count += 1

        if verbose:
            print(
                f"\nWarming Complete! {self.buffer.num_trj()}/{self.buffer.min_num_trj} | total sample_time = {total_sample_time:.2f}s",
                end="",
            )
            print()
        t1 = time.time()
        sample_time = t1 - t0
        return sample_time

    def save_model(self, e: int):
        # Save checkpoint
        if e % self.log_interval == 0:
            self.policy.save_model(self.logger.checkpoint_dirs[5], e)

        # Save best model
        if self.last_reward_mean > self.last_max_reward:
            self.policy.save_model(self.logger.log_dirs[5], e, is_best=True)
            self.last_max_reward = self.last_reward_mean

    def write_log(self, logging_dict: dict, iter_idx: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(int(iter_idx), display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, int(iter_idx))
