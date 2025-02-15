import time
import numpy as np
from copy import deepcopy
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tqdm.auto import trange
from collections import deque
from log.wandb_logger import WandbLogger
from models.policy.hierarchicalController import HC_Controller
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# model-free policy trainer
class HCTrainer:
    def __init__(
        self,
        policy: HC_Controller,
        sampler: OnlineSampler,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        timesteps: int = 1000,
        init_timesteps: int = 0,
        lr_scheduler: torch.optim.lr_scheduler = None,
        log_interval: int = 2,
        grid_type: int = 0,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.evaluator = evaluator

        self.logger = logger
        self.writer = writer

        # training parameters
        self.timesteps = timesteps
        self.init_timesteps = init_timesteps
        self.eval_num = 0
        self.eval_interval = int(self.timesteps / log_interval)
        self.lr_scheduler = lr_scheduler

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.grid_type = grid_type

    def train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # Train loop
        with tqdm(total=self.timesteps, desc=f"HC Training (Timesteps)") as pbar:
            while pbar.n < self.timesteps:
                self.policy.train()
                batch, sample_time = self.sampler.collect_samples(
                    self.policy, grid_type=self.grid_type, is_option=True
                )
                loss_dict, hc_timesteps, update_time = self.policy.learn(batch)

                # Calculate expected remaining time
                pbar.update(hc_timesteps)

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / pbar.n
                remaining_time = avg_time_per_iter * (self.timesteps - pbar.n)

                # Update environment steps and calculate time metrics
                loss_dict["HC/analytics/timesteps"] = pbar.n
                loss_dict["HC/analytics/sample_time"] = sample_time
                loss_dict["HC/analytics/update_time"] = update_time
                loss_dict["HC/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss_dict, step=int(pbar.n + self.init_timesteps))

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                ### Eval Loop
                if pbar.n >= self.eval_interval * (self.eval_num + 1):

                    self.policy.eval()
                    self.eval_num += 1

                    temp_eval_dict, supp_dict = self.evaluator(
                        self.policy,
                        grid_type=self.grid_type,
                    )

                    eval_dict = {}
                    for k in temp_eval_dict.keys():
                        eval_dict["HC/" + k] = temp_eval_dict[k]

                    self.write_log(
                        eval_dict,
                        step=int(pbar.n + self.init_timesteps),
                        eval_log=True,
                    )
                    self.write_image(
                        supp_dict,
                        step=int(pbar.n + self.init_timesteps),
                        log_dir="HC_image/",
                    )
                    self.write_video(
                        supp_dict,
                        step=int(pbar.n + self.init_timesteps),
                        log_dir="HC_video/",
                    )

                    self.last_reward_mean.append(eval_dict["HC/rew_mean"])
                    self.last_reward_std.append(eval_dict["HC/rew_std"])

                    self.save_model(int(pbar.n + self.init_timesteps))

        self.logger.print(
            f"total HC training time: {((time.time() - start_time) / 3600):.2f} hours"
        )

    def write_log(self, logging_dict: dict, step: int, eval_log: bool = False):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, eval_log=eval_log, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, supp_dict: dict, step: int, log_dir: str):
        if supp_dict["path_image"] is not None:
            image_list = [supp_dict["path_image"]]
            path_image_path = log_dir + f"path_image"
            self.logger.write_images(
                step=step, images=image_list, log_dir=path_image_path
            )
        if supp_dict["option_image"] is not None:
            image_list = [supp_dict["option_image"]]
            path_image_path = log_dir + f"option_image"
            self.logger.write_images(
                step=step, images=image_list, log_dir=path_image_path
            )

    def write_video(self, supp_dict: dict, step: int, log_dir: str):
        if supp_dict["path_render"] is not None:
            path_render_path = log_dir + f"path_render"
            self.logger.write_videos(
                step=step, images=supp_dict["path_render"], log_dir=path_render_path
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
            self.policy.save_model(self.logger.log_dirs[2], e, name="HC", is_best=True)
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
