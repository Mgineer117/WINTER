import time
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from tqdm.auto import trange
from collections import deque
from log.wandb_logger import WandbLogger
from models.policy.optionPolicy import OP_Controller
from utils.sampler import OnlineSampler
from utils.buffer import TrajectoryBuffer
from models.evaulators.sf_evaluator import Evaluator


# Custom scheduler logic for different parameter groups
def custom_lr_scheduler(optimizer, timesteps, schedulingtimesteps=1):
    if timesteps % schedulingtimesteps == 0:
        optimizer["ppo"].param_groups[0]["lr"] *= 0.7  # Reduce learning rate for phi


# model-free policy trainer
class OPTrainer:
    def __init__(
        self,
        policy: OP_Controller,
        sampler: OnlineSampler,
        buffer: TrajectoryBuffer,
        logger: WandbLogger,
        writer: SummaryWriter,
        evaluator: Evaluator,
        num_weights: int,
        mode: str,
        timesteps: int = 1e6,
        init_timesteps: int = 0,
        batch_size: int = 1024,
        lr_scheduler: torch.optim.lr_scheduler = None,
        log_interval: int = 10,
        grid_type: int = 0,
    ) -> None:

        # training parameters
        self.num_weights = num_weights
        self.mode = mode
        self.timesteps = timesteps
        self.init_timesteps = init_timesteps
        self.batch_size = batch_size

        self.eval_num = 0
        self.eval_interval = int(self.timesteps / log_interval)

        self.lr_scheduler = lr_scheduler

        # trainable parameters
        self.policy = policy
        self.sampler = sampler
        self.buffers = [deepcopy(buffer) for _ in range(self.num_weights)]
        self.evaluator = evaluator

        # logger
        self.logger = logger
        self.writer = writer

        # initialize the essential training components
        self.last_max_reward = -1e10
        self.std_limit = 0.5
        self.num_env_steps = 0

        self.log_interval = log_interval
        self.grid_type = grid_type

    def train(self) -> dict[str, float]:
        if self.mode == "ppo":
            first_finaltimesteps = self.ppo_train()
        elif self.mode == "sac":
            first_finaltimesteps = self.sac_train()
        else:
            raise NotImplementedError(f"{self.mode} is not implemented")

        return first_finaltimesteps

    def sac_train(self) -> dict[str, float]:
        """DEPRECATED

        Returns:
            Dict[str, float]: _description_
        """
        start_time = time.time()
        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # Train loop
        firstinit_timesteps = self.init_timesteps
        first_finaltimesteps = self.timesteps

        total_iterations = (
            first_finaltimesteps - firstinit_timesteps
        ) * self._step_pertimesteps
        completed_iterations = 0

        sample_time = self.warm_buffer()
        for e in trange(
            firstinit_timesteps, first_finaltimesteps, desc=f"OP SAC timesteps"
        ):
            ### Training loop
            self.policy.train()
            for it in trange(self._step_pertimesteps, desc="Training", leave=False):
                sample_time = 0
                update_time = 0
                policy_loss = []
                for z in trange(self.num_weights, desc=f"Updating Option", leave=False):
                    batch = self.buffers[z].sample(self.batch_size)
                    loss_dict, updateT = self.policy.learn(batch, z)

                    batch, sampleT = self.sampler.collect_samples(
                        self.policy,
                        idx=z,
                        grid_type=self.grid_type,
                        random_init_pos=True,
                    )
                    self.buffers[z].push(batch)

                    update_time += updateT
                    sample_time += sampleT
                    policy_loss.append(loss_dict)

                # Calculate expected remaining time
                completed_iterations += 1
                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / completed_iterations
                remaining_time = avg_time_per_iter * (
                    total_iterations - completed_iterations
                )
                # Logging further info
                loss = self.average_dict_values(policy_loss)
                loss["OP_SAC/sample_time"] = sample_time
                loss["OP_SAC/update_time"] = update_time
                loss["OP_SAC/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss, step=int(e * self._step_pertimesteps + it))

            if e % self.log_interval == 0:
                ### Eval Loop
                self.policy.eval()
                rew_mean = np.zeros((self.num_weights,))
                rew_std = np.zeros((self.num_weights,))
                ln_mean = np.zeros((self.num_weights,))
                ln_std = np.zeros((self.num_weights,))

                for z in trange(self.num_weights, desc=f"Evaluation", leave=False):
                    eval_dict = self.evaluator(
                        self.policy,
                        timesteps=e + 1,
                        iter_idx=int(
                            e * self._step_pertimesteps + self._step_pertimesteps
                        ),
                        idx=z,
                        name1=z,
                        dir_name="OP_SAC",
                        write_log=False,  # since OP needs to write log of average of all options
                        grid_type=self.grid_type,
                    )
                    rew_mean[z] = eval_dict["rew_mean"]
                    rew_std[z] = eval_dict["rew_std"]
                    ln_mean[z] = eval_dict["ln_mean"]
                    ln_std[z] = eval_dict["ln_std"]

                rew_mean = np.mean(rew_mean)
                rew_std = np.mean(rew_std)
                ln_mean = np.mean(ln_mean)
                ln_std = np.mean(ln_std)

                # Manual logging
                eval_dict = {
                    "OP_SAC/eval_rew_mean": rew_mean,
                    "OP_SAC/eval_rew_std": rew_std,
                    "OP_SAC/eval_ln_mean": ln_mean,
                    "OP_SAC/eval_ln_std": ln_std,
                }
                self.evaluator.write_log(
                    eval_dict,
                    iter_idx=int(e * self._step_pertimesteps + self._step_pertimesteps),
                )

                self.last_reward_mean.append(rew_mean)
                self.last_reward_std.append(rew_std)

                self.savemodel(e + 1)

        return first_finaltimesteps

    def ppo_train(self) -> dict[str, float]:
        start_time = time.time()

        self.last_reward_mean = deque(maxlen=3)
        self.last_reward_std = deque(maxlen=3)

        # Train loop
        with tqdm(total=self.timesteps, desc=f"OP Training (Timesteps)") as pbar:
            while pbar.n < self.timesteps:
                self.policy.train()

                update_time = 0
                policy_loss = []

                # Sample batch
                batches, sample_time = self.sampler.collect_samples(
                    self.policy,
                    option_indices=[i for i in range(self.num_weights)],
                    grid_type=self.grid_type,
                    random_init_pos=True,
                )

                for z in trange(self.num_weights, desc=f"Updating Option", leave=False):
                    # Update params
                    loss_dict, op_timesteps, updateT = self.policy.learn(batches[z], z)
                    policy_loss.append(loss_dict)
                    update_time += updateT

                    # Calculate expected remaining time
                    pbar.update(int(op_timesteps / self.num_weights))

                elapsed_time = time.time() - start_time
                avg_time_per_iter = elapsed_time / pbar.n
                remaining_time = avg_time_per_iter * (self.timesteps - pbar.n)

                # Logging further info
                loss = self.average_dict_values(policy_loss)
                loss["OP_PPO/analytics/timesteps"] = pbar.n
                loss["OP_PPO/analytics/sample_time"] = sample_time
                loss["OP_PPO/analytics/update_time"] = update_time
                loss["OP_PPO/analytics/remaining_time (hr)"] = (
                    remaining_time / 3600
                )  # Convert to hours

                self.write_log(loss, step=int(pbar.n + self.init_timesteps))

                if pbar.n >= self.eval_interval * (self.eval_num + 1):
                    ### Eval Loop
                    self.policy.eval()
                    self.eval_num += 1

                    eval_dict_list = []
                    supp_dict_list = []
                    for z in trange(self.num_weights, desc=f"Evaluation", leave=False):
                        eval_dict, supp_dict = self.evaluator(
                            self.policy,
                            idx=z,
                            grid_type=self.grid_type,
                        )
                        eval_dict_list.append(eval_dict)
                        supp_dict_list.append(supp_dict)

                    eval_avg_dict = self.average_dict_values(eval_dict_list)

                    eval_dict = {}
                    for k in eval_avg_dict.keys():
                        eval_dict["OP_PPO/" + k] = eval_avg_dict[k]

                    # Manual logging
                    self.write_log(
                        eval_dict,
                        step=int(pbar.n + self.init_timesteps),
                    )
                    self.write_image(
                        supp_dict_list,
                        step=int(pbar.n + self.init_timesteps),
                        log_dir="OP_PPO_image/",
                    )
                    self.write_video(
                        supp_dict_list,
                        step=int(pbar.n + self.init_timesteps),
                        log_dir="OP_PPO_video/",
                    )

                    self.last_reward_mean.append(eval_dict["OP_PPO/rew_mean"])
                    self.last_reward_std.append(eval_dict["OP_PPO/rew_std"])

                    self.save_model(int(pbar.n + self.init_timesteps))

        self.policy.eval()
        self.logger.print(
            "total OP training time: {:.2f} hours".format(
                (time.time() - start_time) / 3600
            )
        )

        return int(pbar.n + self.init_timesteps)

    def evaluate(self, timesteps: int = 5):
        self.evaluator.log_interval = 1
        firstinit_timesteps = self.init_timesteps
        first_finaltimesteps = self.init_timesteps + timesteps
        ### Eval Loop
        for e in trange(
            firstinit_timesteps, first_finaltimesteps, desc=f"OP Eval timesteps"
        ):
            self.policy.eval()
            rew_mean = np.zeros((self.num_weights,))
            rew_std = np.zeros((self.num_weights,))
            ln_mean = np.zeros((self.num_weights,))
            ln_std = np.zeros((self.num_weights,))
            for z in trange(self.num_weights, desc=f"Evaluation", leave=False):
                eval_dict = self.evaluator(
                    self.policy,
                    timesteps=e + 1,
                    iter_idx=int(e * self._step_pertimesteps + self._step_pertimesteps),
                    idx=z,
                    name1=z,
                    dir_name="OP",
                    write_log=False,  # since OP needs to write log of average of all options
                    grid_type=self.grid_type,
                )
                rew_mean[z] = eval_dict["rew_mean"]
                rew_std[z] = eval_dict["rew_std"]
                ln_mean[z] = eval_dict["ln_mean"]
                ln_std[z] = eval_dict["ln_std"]
                torch.cuda.empty_cache()

            rew_mean = np.mean(rew_mean)
            rew_std = np.mean(rew_std)
            ln_mean = np.mean(ln_mean)
            ln_std = np.mean(ln_std)

            # manual logging
            eval_dict = {
                "OP/eval_rew_mean": rew_mean,
                "OP/eval_rew_std": rew_std,
                "OP/eval_ln_mean": ln_mean,
                "OP/eval_ln_std": ln_std,
            }
            self.evaluator.write_log(
                eval_dict,
                iter_idx=int(e * self._step_pertimesteps + self._step_pertimesteps),
            )

            torch.cuda.empty_cache()

        return timesteps

    def write_log(self, logging_dict: dict, step: int):
        # Logging to WandB and Tensorboard
        self.logger.store(**logging_dict)
        self.logger.write(step, display=False)
        for key, value in logging_dict.items():
            self.writer.add_scalar(key, value, step)

    def write_image(self, supp_dict_list: list, step: int, log_dir: str):
        for i, supp_dict in enumerate(supp_dict_list):
            if supp_dict["path_image"] is not None:
                path_image_path = log_dir + f"path_image_{i}"
                self.logger.write_videos(
                    step=step, images=[supp_dict["path_image"]], log_dir=path_image_path
                )

    def write_video(self, supp_dict_list: list, step: int, log_dir: str):
        for i, supp_dict in enumerate(supp_dict_list):
            if supp_dict["path_render"] is not None:
                path_render_path = log_dir + f"path_render_{i}"
                self.logger.write_videos(
                    step=step, images=supp_dict["path_render"], log_dir=path_render_path
                )

    def warm_buffer(self, verbose=False):
        t0 = time.time()
        for z, buffer in enumerate(self.buffers):
            # make sure there is nothing there
            buffer.wipe()

            # collect enough batch
            count = 0
            total_sample_time = 0
            sample_time = 0
            self.num_env_steps += self.batch_size
            while buffer.num_samples < buffer.minbatch_size:
                batch, sampleT = self.sampler.collect_samples(
                    self.policy, idx=z, grid_type=self.grid_type, random_init_pos=True
                )
                buffer.push(batch)
                sample_time += sampleT
                total_sample_time += sampleT
                if count % 50 == 0:
                    if verbose:
                        print(
                            f"\nWarming buffer {buffer.num_samples}/{buffer.minbatch_size} | sample_time = {sample_time:.2f}s",
                            end="",
                        )
                    sample_time = 0
                count += 1
            if verbose:
                print(
                    f"\nWarming Complete! {buffer.num_samples}/{buffer.minbatch_size} | total sample_time = {total_sample_time:.2f}s",
                    end="",
                )
                print()

        t1 = time.time()
        sample_time = t1 - t0

        return sample_time

    def save_model(self, e):
        # save checkpoint
        self.policy.save_model(self.logger.checkpoint_dirs[1], e)

        # save the best model
        if (
            np.mean(self.last_reward_mean) > self.last_max_reward
            and np.mean(self.last_reward_std) <= self.std_limit
        ):
            self.policy.save_model(self.logger.log_dirs[1], e, is_best=True)
            self.last_max_reward = np.mean(self.last_reward_mean)

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
