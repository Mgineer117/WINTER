import time
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing

from typing import Optional, Dict, List
from tqdm.auto import trange
from log.wandb_logger import WandbLogger
from models.policy.base_policy import BasePolicy
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# model-free policy trainer
class SFTrainer:
    def __init__(
        self,
        policy: BasePolicy,
        sampler: OnlineSampler,
        buffer: TrajectoryBuffer,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        scheduler: torch.optim.lr_scheduler,
        ### Parmaterers ###
        epoch: int = 1000,
        init_epoch: int = 0,
        step_per_epoch: int = 1000,
        eval_episodes: int = 10,
        log_interval: int = 2,
        post_process: str = "nonzero_rewards",
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        self.scheduler = scheduler

        # training parameters
        self._init_epoch = init_epoch
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch

        self._eval_episodes = eval_episodes

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.last_min_std = 0.0
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.post_process = post_process
        self.grid_type = grid_type

    def train(self) -> Dict[str, float]:
        start_time = time.time()

        # Calculate the total number of iterations for both training phases
        total_iterations = (self._epoch - self._init_epoch) * self._step_per_epoch
        current_iteration = 0

        # Warm buffer
        sample_time = self.warm_buffer(post_process=self.post_process, minimum=True)

        init_epoch = self._init_epoch
        final_epoch = self._epoch

        for e in trange(init_epoch, final_epoch, desc="SF Phi Epoch"):
            ### Training loop
            self.policy.train()
            for it in trange(self._step_per_epoch, desc="Training", leave=False):
                # Track iteration progress
                current_iteration += 1
                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / current_iteration
                remaining_time = avg_time_per_iter * (
                    total_iterations - current_iteration
                )

                # Training step
                loss, update_time = self.policy.learn(self.buffer)
                loss["SF/sample_time"] = sample_time
                loss["SF/update_time"] = update_time
                loss["SF/lr"] = self.policy.feature_optims.param_groups[0]["lr"]
                loss["SF/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Log remaining time

                self.write_log(loss, step=int(e * self._step_per_epoch + it))
                sample_time = 0

            if self.scheduler is not None:
                self.scheduler.step()

            if not self.buffer.full:
                batch, sample_time = self.sampler.collect_samples(
                    self.policy, grid_type=self.grid_type, random_init_pos=True
                )
                self.buffer.push(batch)

            ### Eval
            if (e + 1) % self.log_interval == 0:
                self.policy.eval()
                eval_dict = self.policy.evaluate(self.buffer)
                self.write_image(
                    eval_dict=eval_dict,
                    step=int(e * self._step_per_epoch + it),
                    log_dir="SF/",
                )
                self.save_model(e + 1)
                # self.evaluator(
                #     self.policy,
                #     epoch=e,
                #     iter_idx=int(e * self._step_per_epoch + self._step_per_epoch),
                #     dir_name="SF",
                #     grid_type=self.grid_type,
                # )

        self.buffer.wipe()
        torch.cuda.empty_cache()

        self.policy.eval()
        self.logger.print(
            "total SF training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

        return final_epoch

    def save_model(self, e):
        # save checkpoint
        self.policy.save_model(self.logger.checkpoint_dirs[0], e)

    def warm_buffer(self, post_process: str | None, minimum: bool):
        t0 = time.time()

        if minimum:
            threshold = self.buffer.min_batch_size
        else:
            threshold = self.buffer.max_batch_size

        # make sure there is nothing there
        self.buffer.wipe()

        # collect enough batch
        count = 0
        total_sample_time = 0
        sample_time = 0
        while self.buffer.num_samples < threshold:
            batch, sampleT = self.sampler.collect_samples(
                self.policy, grid_type=self.grid_type, random_init_pos=True
            )
            self.buffer.push(batch, post_process=post_process)
            sample_time += sampleT
            total_sample_time += sampleT
            if count % 25 == 0:
                print(
                    f"\nWarming buffer with {post_process} {self.buffer.num_samples}/{threshold} | sample_time = {sample_time:.2f}s",
                    end="",
                )
                sample_time = 0
            count += 1
        print(
            f"\nWarming Complete! {self.buffer.num_samples}/{threshold} | total sample_time = {total_sample_time:.2f}s",
            end="",
        )
        print()
        t1 = time.time()
        sample_time = t1 - t0
        return sample_time

    def write_log(self, logging_dict: dict, step: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, eval_dict: dict, step: int, log_dir: str):
        if eval_dict["ground_truth"][0] is not None:
            true_path = log_dir + "true"
            self.logger.write_images(
                step=step, images=eval_dict["ground_truth"], log_dir=true_path
            )
        if eval_dict["prediction"][0] is not None:
            pred_path = log_dir + "pred"
            self.logger.write_images(
                step=step, images=eval_dict["prediction"], log_dir=pred_path
            )
        if eval_dict["reward_plot"][0] is not None:
            reward_path = log_dir + "reward_plot"
            self.logger.write_images(
                step=step,
                images=eval_dict["reward_plot"],
                log_dir=reward_path,
            )
        if eval_dict["feature_plot"][0] is not None:
            feature_path = log_dir + "feature_plot"
            self.logger.write_images(
                step=step,
                images=eval_dict["feature_plot"],
                log_dir=feature_path,
            )
