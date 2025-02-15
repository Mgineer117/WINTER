import torch
import torch.nn as nn
import cv2
import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from models.evaulators.base_evaluator import DotDict

from utils import *
from utils.call_env import call_env

import wandb

wandb.require("core")


def remove_dir(dir_path):
    # Iterate over all the files and subdirectories
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # Delete all files
        for name in files:
            os.remove(os.path.join(root, name))
        # Delete all directories
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    # Finally, delete the main directory
    os.rmdir(dir_path)


def run_loop(env, option_vals, options, args):
    # for i in [0, 4, 5, 6, 9]:
    if args.env_name == "FourRooms":
        grid, pos, loc = get_grid_tensor(env, grid_type=args.grid_type)

        save_path = f"RewardMap/FourRooms"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        else:
            remove_dir(save_path)
            os.mkdir(save_path)

        # do reward Plot
        plotter.plotRewardMap(
            feaNet=sf_network.feaNet,
            S=option_vals,
            V=options,
            feature_dim=args.sf_dim,
            algo_name=args.algo_name,
            grid_tensor=grid,
            coords=pos,
            loc=loc,
            dir=save_path,
            device=args.device,
        )
    elif args.env_name == "LavaRooms":
        for i in range(10):
            save_path = f"RewardMap/LavaRooms/{str(i)}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            else:
                remove_dir(save_path)
                os.mkdir(save_path)

            grid, pos, agent_pos = get_grid_tensor(env, grid_type=args.grid_type)
            # do reward Plot
            plotter.plotRewardMap(
                feaNet=sf_network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,
                algo_name=args.algo_name,
                grid_tensor=grid,
                coords=pos,
                agent_pos=agent_pos,
                dir=save_path,
                device=args.device,
            )

    elif args.env_name == "CtF1v1" or args.env_name == "CtF1v2":
        # prepare the grid
        obs, _ = env.reset(seed=args.grid_type)
        grid_tensor = obs["observation"]
        env.close()

        agent_pos = np.where(grid_tensor[:, :, 1] == 1)
        enemy_pos = np.where(grid_tensor[:, :, 1] == 2)

        grid_tensor[agent_pos[0], agent_pos[1], 1] = 0
        grid_tensor[agent_pos[0], agent_pos[1], 2] = 0

        grid_tensor[enemy_pos[0], enemy_pos[1], 1] = 0
        grid_tensor[enemy_pos[0], enemy_pos[1], 2] = 0

        x_coords, y_coords = np.where(
            (grid_tensor[:, :, 0] != 0)
            & (grid_tensor[:, :, 1] != 3)
            & (grid_tensor[:, :, 1] != 4)
        )  # find idx where not wall

        random_index = random.sample(range(len(x_coords)), 1)
        for x, y in zip(x_coords, y_coords):
            grid = grid_tensor.copy()
            # enemy assignment
            grid[x, y, 1] = 2
            grid[x, y, 2] = 2
            if args.env_name == "CtF1v2":
                grid[x_coords[random_index], y_coords[random_index], 1] = 2
                grid[x_coords[random_index], y_coords[random_index], 2] = 2

            # find idx where not wall and red agent
            pos = np.where(
                (grid[:, :, 0] != 0)
                & (grid[:, :, 1] != 2)
                & (grid[:, :, 1] != 3)
                & (grid[:, :, 1] != 4)
            )

            agent_pos = np.full((2 * args.agent_num,), np.nan, dtype=np.float32)
            agent_pos[2] = x
            agent_pos[3] = y
            if args.env_name == "CtF1v2":
                agent_pos[4] = x_coords[random_index]
                agent_pos[5] = y_coords[random_index]

            # prepare the path
            save_path = f"RewardMap/CtF/{str(x)}_{str(y)}"
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            else:
                remove_dir(save_path)
                os.mkdir(save_path)
            # do reward Plot
            plotter.plotRewardMap2(
                feaNet=sf_network.feaNet,
                S=option_vals,
                V=options,
                feature_dim=args.sf_dim,
                algo_name=args.algo_name,
                grid_tensor=grid,
                coords=pos,
                agent_pos=agent_pos,
                dir=save_path,
                device=args.device,
            )


if __name__ == "__main__":
    algo_name = "SNAC+++"
    env_name = "CtF1v2"
    seed = 1
    # call json
    model_dir = f"log/eval_log/model_for_eval/{env_name}/{algo_name}/"
    with open(model_dir + "config.json", "r") as json_file:
        config = json.load(json_file)
    args = DotDict(config)
    args.algo_name = algo_name
    args.env_name = env_name
    args.num_vector = 16
    args.division_rate = 0.25
    args.device = torch.device("cpu")

    print(f"Algo name: {args.algo_name}")
    print(f"Env name: {args.env_name}")

    # call sf
    args.import_sf_model = True
    sf_network = call_sfNetwork(args)
    plotter = Plotter(
        grid_size=args.grid_size,
        img_tile_size=args.img_tile_size,
        device=args.device,
    )

    # call env
    env = call_env(args)
    save_dim_to_args(env, args)  # given env, save its state and action dim

    sampler = OnlineSampler(
        training_envs=env,
        state_dim=args.s_dim,
        feature_dim=args.sf_dim,
        action_dim=args.a_dim,
        hc_action_dim=args.num_vector + 1,
        agent_num=args.agent_num,
        min_cover_option_length=args.min_cover_option_length,
        episode_len=args.episode_len,
        batch_size=args.batch_size,
        min_batch_for_worker=args.min_batch_for_worker,
        cpu_preserve_rate=args.cpu_preserve_rate,
        num_cores=args.num_cores,
        gamma=args.gamma,
        verbose=False,
    )

    option_vals, options, _ = get_eigenvectors(
        env,
        sf_network,
        sampler,
        plotter,
        args,
        draw_map=False,
    )

    obs, _ = env.reset(seed=args.grid_type)
    grid_tensor = obs["observation"]

    run_loop(env, option_vals, options, args)
