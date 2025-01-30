import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

from utils.plotter import Plotter
from log.wandb_logger import WandbLogger
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as multiprocessing

from typing import Any, DefaultDict, Dict, List, Optional, Tuple


class DotDict(dict):
    """A dictionary subclass that supports access via the dot operator."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not found in the dictionary")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not found in the dictionary")


def check_all_devices(module):
    devices = {param.device for param in module.parameters()}  # Get all unique devices
    return devices


class Evaluator:
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env,
        testing_env=None,
        eval_ep_num: int = 1,
    ):
        # store envs for evaluation
        if testing_env is not None:
            self.envs = training_env + testing_env
        else:
            if not isinstance(training_env, List):
                self.envs = [training_env]
            else:
                self.envs = training_env

        self.logger = logger
        self.writer = writer

        self.eval_ep_num = eval_ep_num

    def __call__(
        self,
        policy: nn.Module,
        idx: int = None,
        grid_type: int = 0,
    ) -> Dict[str, List[float]]:
        """
        policy: decision-maker
        epoch: epoch for rendering logging
        iter_idx: epoch*iter: for wandb or tensorboard logging
        idx: option index to test
        dir_name: dir name for logging
        name1:str: for naming, indexing purpose at one's taste
        name2:str: for naming, indexing purpose at one's taste
        name3:str: for naming, indexing purpose at one's taste
        """
        if isinstance(policy, List):
            policy_device = policy[0].device
            for p in policy:
                p.to_device(torch.device("cpu"))
        else:
            policy_device = policy.device
            policy.to_device(torch.device("cpu"))
            all_devices = check_all_devices(policy)

        eval_dict_list = []
        supp_dict_list = []

        # Using Multiprocessing crashes graphic rendering process, so we iterate all envs one by one
        for i, env in enumerate(self.envs):
            eval_dict, supp_dict = self.eval_loop(
                env,
                policy,
                idx=idx,
                grid_type=grid_type,
            )
            eval_dict_list.append(eval_dict)
            supp_dict_list.append(supp_dict)

        eval_dict = self.average_dict_values(eval_dict_list)
        supp_dict = self.average_dict_values(supp_dict_list)

        if isinstance(policy, list):
            for p in policy:
                p.to_device(policy_device)
        else:
            policy.to_device(policy_device)

        return eval_dict, supp_dict

    def set_any_seed(self, grid_type, seed):
        """
        This saves current seed info and calls after stochastic action selection.
        -------------------------------------------------------------------------
        This is to introduce the stochacity in each multiprocessor.
        Without this, the samples from each multiprocessor will be same since the seed was fixed
        """

        # Set the temporary seed
        temp_seed = grid_type + seed
        torch.manual_seed(temp_seed)
        np.random.seed(temp_seed)
        random.seed(temp_seed)

    def average_dict_values(self, dict_list):
        if not dict_list:
            return {}

        if len(dict_list) == 1:
            return dict_list[0]
        else:
            # Initialize a dictionary to hold the sum of values for each key
            sum_dict = {key: 0 for key in dict_list[0].keys()}

            # Iterate over each dictionary in the list
            for d in dict_list:
                for key, value in d.items():
                    sum_dict[key] += value

            # Calculate the average for each key
            avg_dict = {
                key: sum_val / len(dict_list) for key, sum_val in sum_dict.items()
            }

            return avg_dict
