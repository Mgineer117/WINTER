import pickle
import wandb
import torch
import json
import os
import sys

sys.path.append("../SNAC")

import gymnasium as gym
from gym_multigrid.envs.fourrooms import FourRooms
from models.evaulators.sf_evaluator import SF_Evaluator
from models.evaulators.base_evaluator import DotDict
from utils.call_network import call_sfNetwork
from utils.utils import seed_all
from utils.plotter import Plotter
from torch.utils.tensorboard import SummaryWriter
from log.base_logger import BaseLogger

wandb.require("core")


def train(eval_ep_num=10):
    model_dir = "log/eval_log/model_for_eval/"
    with open(model_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    seed = 1
    # call env
    # env = FourRooms(
    #     grid_size=(args.grid_size, args.grid_size),
    #     agent_view_size=args.agent_view_size,
    #     max_steps=args.episode_len,
    #     render_mode="rgb_array",
    # )

    env = gym.make(
        "MiniGrid-FourRooms-v0",
        render_mode="rgb_array",
        max_steps=args.episode_len,
        agent_view_size=args.agent_view_size,
    )
    # import pre-trained model before defining actual models
    print("Loading previous model parameters....")
    args.import_model = True
    policy = call_sfNetwork(args)

    # Call loggers
    logdir = "log/eval_log/result/"
    logger = BaseLogger(logdir, name=args.name)
    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    if not os.path.exists(tensorboard_path):
        os.mkdir(tensorboard_path)
    writer = SummaryWriter(log_dir=tensorboard_path)

    sf_path = logger.checkpoint_dirs[0]  # SF checkpoint b/c plotter will only be used

    print(f"Saving Directory = {logdir + args.name}")
    print(f"Result is an average of {eval_ep_num} episodes")

    SF_evaluator = SF_Evaluator(
        logger=logger,
        writer=writer,
        training_env=env,
        dir=sf_path,
        log_interval=args.log_interval,
    )

    SF_evaluator(
        policy,
        epoch=0,
        iter_idx=int(0),
        dir_name="SF",
        seed=seed,
    )
    print("Evaluation is done!")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
