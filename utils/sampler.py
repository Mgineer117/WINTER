import os
import random
import time
import math

import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
import numpy as np
import cv2
from math import floor, ceil

from datetime import date
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

today = date.today()


class Base:
    def __init__():
        pass

    def get_reset_data(self, batch_size, init="nan"):
        """
        We create a initialization batch to avoid the daedlocking.
        The remainder of zero arrays will be cut in the end.
        np.nan makes it easy to debug
        """
        if init == "zero":
            data = dict(
                states=np.zeros(((batch_size,) + self.state_dim), dtype=np.float32),
                next_states=np.zeros(
                    ((batch_size,) + self.state_dim), dtype=np.float32
                ),
                actions=np.zeros((batch_size, self.action_dim), dtype=np.float32),
                option_actions=np.zeros(
                    (batch_size, self.hc_action_dim), dtype=np.int8
                ),
                rewards=np.zeros((batch_size, 1), dtype=np.float32),
                terminals=np.zeros((batch_size, 1), dtype=np.int8),
                logprobs=np.zeros((batch_size, 1), dtype=np.float32),
                entropys=np.zeros((batch_size, 1), dtype=np.float32),
            )
        elif init == "nan":
            data = dict(
                states=np.full(
                    ((batch_size,) + self.state_dim), np.nan, dtype=np.float32
                ),
                next_states=np.full(
                    ((batch_size,) + self.state_dim), np.nan, dtype=np.float32
                ),
                actions=np.full(
                    (batch_size, self.action_dim), np.nan, dtype=np.float32
                ),
                option_actions=np.full(
                    (batch_size, self.hc_action_dim), np.nan, dtype=np.int8
                ),
                rewards=np.full((batch_size, 1), np.nan, dtype=np.float32),
                terminals=np.full((batch_size, 1), np.nan, dtype=np.int8),
                logprobs=np.full((batch_size, 1), np.nan, dtype=np.float32),
                entropys=np.full((batch_size, 1), np.nan, dtype=np.float32),
            )
        else:
            NotImplementedError("Not implemented")

        return data

    def set_any_seed(self, seed, pid):
        """
        This saves current seed info and calls after stochastic action selection.
        -------------------------------------------------------------------------
        This is to introduce the stochacity in each multiprocessor.
        Without this, the samples from each multiprocessor will be same since the seed was fixed
        """

        temp_seed = seed + pid

        # Set the temporary seed
        torch.manual_seed(temp_seed)
        np.random.seed(temp_seed)
        random.seed(temp_seed)

    def collect_samples(
        self,
        policy,
        option_indices: list | None = [None],
        grid_type: int = 0,
        random_init_pos: bool = False,
        deterministic: bool = False,
        is_option: bool = False,
    ):
        """
        All sampling and saving to the memory is done in numpy.
        return: dict() with elements in numpy
        """
        t_start = time.time()

        policy_device = policy.device
        policy.to_device(torch.device("cpu"))

        # Select appropriate sampler function
        sample_fn = (
            self.collect_trajectory4Option if is_option else self.collect_trajectory
        )

        idx_idx = 0
        worker_idx = 0

        # Use persistent queue from self.manager
        if not hasattr(self, "manager"):
            self.manager = multiprocessing.Manager()
            self.queue = self.manager.Queue()

        queue = self.queue

        # Iterate over rounds
        for round_number in range(self.rounds):
            processes = []
            indices = (
                option_indices[idx_idx:]
                if round_number == self.rounds - 1
                else option_indices[
                    idx_idx : idx_idx + self.num_idx_per_round[round_number]
                ]
            )

            # Iterate over indices
            for idx in indices:
                for i in range(self.num_worker_per_idx):
                    if worker_idx == self.total_num_worker - 1:
                        # Main thread process
                        try:
                            memory = sample_fn(
                                worker_idx,
                                None,
                                self.env,
                                policy,
                                idx,
                                grid_type,
                                random_init_pos,
                                seed=i,
                                deterministic=deterministic,
                            )
                        except Exception as e:
                            print(f"Main thread worker {worker_idx} failed: {e}")
                            memory = None
                    else:
                        # Sub-thread process
                        worker_args = (
                            worker_idx,
                            queue,
                            self.env,
                            policy,
                            idx,
                            grid_type,
                            random_init_pos,
                            i,
                            deterministic,
                        )
                        p = multiprocessing.Process(target=sample_fn, args=worker_args)
                        processes.append(p)
                        p.start()

                    worker_idx += 1

                if worker_idx % self.req_num_workers == 0:
                    idx_idx += 1

            # Ensure all workers finish before collecting data
            for p in processes:
                p.join()

        # Include worker memories in one list
        worker_memories = [None] * worker_idx
        while not queue.empty():
            try:
                pid, worker_memory = queue.get(timeout=2)
                worker_memories[pid] = worker_memory
            except Exception as e:
                print(f"Queue retrieval error: {e}")

        worker_memories[-1] = memory  # Add main thread memory

        # Classify the batch according to the option index
        if self.num_options == 1:
            memory = {}
            for worker_memory in worker_memories:
                if worker_memory is None:
                    continue
                for key in worker_memory:
                    if key in memory:
                        memory[key] = np.concatenate(
                            (memory[key], worker_memory[key]), axis=0
                        )
                    else:
                        memory[key] = worker_memory[key]

            # Truncate to batch size
            for k, v in memory.items():
                memory[k] = v[: self.batch_size]

        else:
            memory_dict = {i: [] for i in range(len(option_indices))}
            for i, worker_memory in enumerate(worker_memories):
                if worker_memory is None:
                    continue
                option_index = i // self.num_worker_per_idx
                memory_dict[option_index].append(worker_memory)

            memory = {}
            for i, batch_list in memory_dict.items():
                stacked_dict = {}
                for batch in batch_list:
                    for key, value in batch.items():
                        if key in stacked_dict:
                            stacked_dict[key] = np.concatenate(
                                (stacked_dict[key], value), axis=0
                            )
                        else:
                            stacked_dict[key] = value

                # Truncate to batch size
                for k, v in stacked_dict.items():
                    stacked_dict[k] = v[: self.batch_size]
                memory[i] = stacked_dict

        policy.to_device(policy_device)
        t_end = time.time()

        return memory, t_end - t_start


class OnlineSampler(Base):
    def __init__(
        self,
        env,
        state_dim: tuple,
        action_dim: int,
        hc_action_dim: int,
        min_option_length: int,
        num_options: int,
        episode_len: int,
        batch_size: int,
        min_batch_for_worker: int = 1024,
        cpu_preserve_rate: float = 0.95,
        num_cores: int | None = None,
        gamma: float = 0.99,
        verbose: bool = True,
    ) -> None:
        super(Base, self).__init__()
        """
        This computes the ""very"" appropriate parameter for the Monte-Carlo sampling
        given the number of episodes and the given number of cores the runner specified.
        ---------------------------------------------------------------------------------
        Rounds: This gives several rounds when the given sampling load exceeds the number of threads
        the task is assigned. 
        This assigned appropriate parameters assuming one worker work with 2 trajectories.
        """

        # dimensional params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hc_action_dim = hc_action_dim
        self.gamma = gamma

        # Misc params
        self.min_option_length = min_option_length

        # sampling params
        self.num_options = num_options
        self.episode_len = episode_len
        self.min_batch_for_worker = min_batch_for_worker
        self.thread_batch_size = self.min_batch_for_worker + 2 * self.episode_len
        self.batch_size = batch_size

        self.env = env

        # Preprocess for multiprocessing to avoid CPU overscription and deadlock
        self.cpu_preserve_rate = cpu_preserve_rate
        self.temp_cores = floor(multiprocessing.cpu_count() * self.cpu_preserve_rate)
        self.num_cores = num_cores if num_cores is not None else self.temp_cores

        (
            num_workers_per_round,
            num_idx_per_round,
            req_num_workers,
            rounds,
        ) = self.calculate_workers_and_rounds()

        self.req_num_workers = req_num_workers
        self.num_workers_per_round = num_workers_per_round
        self.total_num_worker = sum(self.num_workers_per_round)
        self.num_idx_per_round = num_idx_per_round
        self.num_worker_per_idx = int(self.total_num_worker / self.num_options)

        self.rounds = rounds
        if verbose:
            print("====================")
            print("Sampling Parameters:")
            print("====================")
            print(
                f"Cores (usage)/(given)     : {self.num_workers_per_round}/{self.num_cores} out of {multiprocessing.cpu_count()}"
            )
            print(f"# Indices each Round    : {self.num_idx_per_round}")
            print(f"Total number of Worker  : {self.total_num_worker}")
            print(f"Max. batch size         : {self.thread_batch_size}")

        # enforce one thread for each worker to avoid CPU overscription.
        torch.set_num_threads(1)

    def initialize(
        self,
        batch_size: int,
        num_option: int | None,
        min_batch_for_worker: int,
        verbose: bool = True,
    ):
        # sampling params
        self.min_batch_for_worker = min_batch_for_worker
        self.num_options = num_option
        self.thread_batch_size = self.min_batch_for_worker + 2 * self.episode_len
        self.batch_size = batch_size

        (
            num_workers_per_round,
            num_idx_per_round,
            req_num_workers,
            rounds,
        ) = self.calculate_workers_and_rounds()

        self.req_num_workers = req_num_workers
        self.num_workers_per_round = num_workers_per_round
        self.total_num_worker = sum(self.num_workers_per_round)
        self.num_idx_per_round = num_idx_per_round
        self.num_worker_per_idx = int(self.total_num_worker / self.num_options)

        self.rounds = rounds
        if verbose:
            print("====================")
            print("Sampling Parameters:")
            print("====================")
            print(
                f"Cores (usage)/(given)     : {self.num_workers_per_round}/{self.num_cores} out of {multiprocessing.cpu_count()}"
            )
            print(f"# Indices each Round    : {self.num_idx_per_round}")
            print(f"Total number of Worker  : {self.total_num_worker}")
            print(f"Max. batch size         : {self.thread_batch_size}")

        # enforce one thread for each worker to avoid CPU overscription.
        torch.set_num_threads(1)

    def calculate_workers_and_rounds(self):
        """
        Calculate the number of workers and rounds for multiprocessing training.

        Returns:
            num_worker_per_round (list): Number of workers per round.
            num_idx_per_round (list): Number of indices per round.
            rounds (int): Total number of rounds.
        """
        # Calculate required number of workers
        req_num_workers = ceil(self.batch_size / self.min_batch_for_worker)
        total_num_workers = req_num_workers * self.num_options

        if total_num_workers > self.num_cores:
            # Available cores per index
            avail_core_per_idx = self.num_cores // self.num_options

            # Calculate the number of workers per round per index
            max_workers_per_round = avail_core_per_idx
            num_worker_per_round = []
            num_idx_per_round = []
            rounds = ceil(total_num_workers / self.num_cores)

            remaining_workers = total_num_workers

            for _ in range(rounds):
                workers_this_round = min(remaining_workers, self.num_cores)
                num_worker_per_round.append(workers_this_round)

                # Calculate the number of indices for this round
                indices_this_round = ceil(workers_this_round / req_num_workers)
                num_idx_per_round.append(indices_this_round)

                remaining_workers -= workers_this_round
        else:
            # All workers can run in a single round
            num_worker_per_round = [total_num_workers]
            num_idx_per_round = [self.num_options]
            rounds = 1

        return num_worker_per_round, num_idx_per_round, req_num_workers, rounds

    def collect_trajectory(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        idx: int | None = None,
        grid_type: int = 0,
        random_init_pos: bool = False,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        # estimate the batch size to hava a large batch
        data = self.get_reset_data(batch_size=self.thread_batch_size)  # allocate memory

        # If no seed is given, generate one
        if seed is None:
            seed = random.randint(0, 1_000_000)

        if queue is not None:
            # Apply different seeds for multiprocessor's action stochacity
            self.set_any_seed(seed, pid)

        current_step = 0
        while current_step < self.min_batch_for_worker:
            # env initialization
            if random_init_pos:
                options = {"random_init_pos": True}
            else:
                options = {"random_init_pos": False}

            obs, _ = env.reset(
                seed=grid_type,
                options=options,
            )

            for t in range(self.episode_len):
                with torch.no_grad():
                    a, metaData = policy(obs, idx, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze() if a.shape[-1] > 1 else [a.item()]

                    # env stepping
                    next_obs, rew, term, trunc, infos = env.step(a)
                    trunc = True if (t + 1) == self.episode_len else False
                    done = term or trunc

                # saving the data
                data["states"][current_step + t] = obs["observation"]
                data["next_states"][current_step + t] = next_obs["observation"]
                data["actions"][current_step + t] = a
                data["rewards"][current_step + t] = rew
                data["terminals"][current_step + t] = done
                data["logprobs"][current_step + t] = (
                    metaData["logprobs"].detach().numpy()
                )
                data["entropys"][current_step + t] = (
                    metaData["entropy"].detach().numpy()
                )

                if done:
                    # clear log
                    current_step += t + 1
                    break

                obs = next_obs

        for k in data:
            data[k] = data[k][:current_step]
        if queue is not None:
            queue.put([pid, data])
        else:
            return data

    def collect_trajectory4Option(
        self,
        pid,
        queue,
        env,
        policy: nn.Module,
        idx: int = None,
        grid_type: int = 0,
        random_init_pos: bool = False,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        # estimate the batch size to hava a large batch
        data = self.get_reset_data(batch_size=self.thread_batch_size)  # allocate memory

        # For each episode, apply different seed for stochasticity
        if seed is None:
            seed = random.randint(0, 1_000_000)

        def env_step(a):
            next_obs, rew, term, trunc1, infos = env.step(a)

            self.external_t += 1
            trunc2 = True if (self.external_t + 1) == self.episode_len else False
            done = term or trunc1 or trunc2

            return next_obs, rew, done, infos

        current_step = 0
        while current_step < self.min_batch_for_worker:
            # env initialization
            if random_init_pos:
                options = {"random_init_pos": True}
            else:
                options = {"random_init_pos": False}
            obs, _ = env.reset(seed=grid_type, options=options)

            self.external_t = 0
            for t in range(self.episode_len):

                with torch.no_grad():
                    # sample action
                    a, metaData = policy(obs, idx, deterministic=deterministic)
                    a = a.cpu().numpy().squeeze() if a.shape[-1] > 1 else [a.item()]

                ### Create an Option Loop
                if metaData["is_option"]:
                    next_obs, rew, done, infos = env_step(a)
                    if not done:
                        if metaData["is_hc_controller"]:
                            for o_t in range(1, self.min_option_length):
                                # env stepping
                                with torch.no_grad():
                                    option_a, option_dict = policy(
                                        next_obs,
                                        metaData["z_argmax"],
                                        deterministic=deterministic,
                                    )
                                    option_a = option_a.cpu().numpy().squeeze()

                                next_obs, op_rew, done, infos = env_step(option_a)
                                rew += self.gamma**o_t * op_rew
                                if done or option_dict["option_termination"]:
                                    break
                        else:
                            o_t = 1
                            option_termination = False
                            while not option_termination:
                                # env stepping
                                with torch.no_grad():
                                    option_a, option_dict = policy(
                                        next_obs,
                                        metaData["z_argmax"],
                                        deterministic=deterministic,
                                    )
                                    option_a = option_a.cpu().numpy().squeeze()

                                next_obs, op_rew, done, infos = env_step(option_a)
                                option_termination = policy.predict_option_termination(
                                    next_obs, metaData["z_argmax"]
                                )
                                rew += self.gamma**o_t * op_rew
                                o_t += 1
                                if done:
                                    break
                else:
                    ### Conventional Loop
                    next_obs, rew, done, infos = env_step(a)

                # saving the data
                data["states"][current_step + t] = obs["observation"]
                data["next_states"][current_step + t] = next_obs["observation"]
                data["actions"][current_step + t] = a
                data["option_actions"][current_step + t] = metaData["z"]
                data["rewards"][current_step + t] = rew
                data["terminals"][current_step + t] = done
                data["logprobs"][current_step + t] = (
                    metaData["logprobs"].detach().numpy()
                )
                data["entropys"][current_step + t] = (
                    metaData["entropy"].detach().numpy()
                )

                if done:
                    # clear log
                    current_step += t + 1
                    break

                obs = next_obs

        for k in data:
            data[k] = data[k][:current_step]

        if queue is not None:
            queue.put([pid, data])
        else:
            return data
