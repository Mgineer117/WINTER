import cv2
import os
import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.plotter import Plotter
from log.wandb_logger import WandbLogger
from models.evaulators.base_evaluator import Evaluator
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


def compute_categorical_entropy(indices, num_categories):
    """
    Computes the categorical distribution and its entropy for a given list of indices.

    Args:
        indices (list or np.ndarray): List of category indices (0 to num_categories-1).
        num_categories (int): Total number of categories (default is 8).

    Returns:
        tuple: A tuple (distribution, entropy_value), where
            - distribution (np.ndarray): The normalized distribution over categories.
            - entropy_value (float): The entropy of the distribution.
    """
    # Flatten the list of arrays into a single array
    indices_flat = np.array(indices)

    # Count occurrences of each category
    counts = np.bincount(indices_flat, minlength=num_categories)

    # Normalize to get the categorical distribution
    distribution = counts / counts.sum()

    # Compute the entropy
    entropy_value = entropy(distribution, base=2)  # Use base-2 for bits

    return distribution, entropy_value


class OC_Evaluator(Evaluator):
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env: gym.Env,
        plotter: Plotter,
        testing_env=None,
        dir: str = None,
        gridPlot: bool = True,
        renderPlot: bool = False,
        render_fps: int = 10,
        eval_ep_num: int = 1,
        log_interval: int = 1,
    ):
        super().__init__(
            logger=logger,
            writer=writer,
            training_env=training_env,
            testing_env=testing_env,
            eval_ep_num=eval_ep_num,
            log_interval=log_interval,
        )
        self.plotter = plotter
        self.render_fps = render_fps

        if dir is not None:
            if gridPlot:
                self.gridPlot = True
                self.gridDir = os.path.join(dir, "grid")
                os.mkdir(self.gridDir)
                self.path = []
            else:
                self.gridPlot = False
            if renderPlot:
                self.renderPlot = True
                self.renderDir = os.path.join(dir, "render")
                os.mkdir(self.renderDir)
                self.recorded_frames = []
            else:
                self.renderPlot = False
        else:
            self.gridPlot = False
            self.renderPlot = False

    def eval_loop(
        self,
        env,
        policy: nn.Module,
        epoch: int,
        idx: int = None,
        name1: str = None,
        name2: str = None,
        name3: str = None,
        grid_type: int = 0,
        seed: int = None,
        queue=None,
    ) -> Dict[str, List[float]]:
        ep_buffer = []
        if queue is not None:
            self.set_any_seed(grid_type, seed)

        successes = np.zeros((self.eval_ep_num,))
        for num_episodes in range(self.eval_ep_num):
            self.update_render_criteria(epoch, num_episodes)

            # logging initialization
            ep_reward, ep_length = 0, 0

            # env initialization
            options = {"random_init_pos": False}
            s, _ = env.reset(seed=grid_type, options=options)

            if self.gridCriteria:
                self.init_grid(env)

            option_indices = {"x": [], "y": []}
            done = False
            t = 0
            while not done:
                with torch.no_grad():
                    a, metaData = policy(s, idx, deterministic=True)
                    a = a.cpu().numpy().squeeze() if a.shape[-1] > 1 else [a.item()]

                # Update the grid
                if self.gridCriteria:
                    self.get_agent_pos(env)

                ns, rew, term, trunc, infos = env.step(a)
                done = term or trunc

                option_termination = False
                step_count = 1
                while not (done or option_termination):
                    with torch.no_grad():
                        option_a, _ = policy(
                            ns, metaData["z_argmax"], deterministic=True
                        )
                        option_a = (
                            option_a.cpu().numpy().squeeze()
                            if a.shape[-1] > 1
                            else [a.item()]
                        )

                    # Update the grid
                    if self.gridCriteria:
                        self.get_agent_pos(env)

                    ns, op_rew, term, trunc, infos = env.step(option_a)

                    option_termination = policy.predict_option_termination(
                        ns, metaData["z_argmax"]
                    )
                    done = term or trunc

                    rew += 0.99**step_count * op_rew
                    step_count += 1

                s = ns

                if "success" in infos:
                    successes[num_episodes] = np.maximum(
                        successes[num_episodes], infos["success"]
                    )
                ep_reward += rew
                ep_length += 1
                option_indices["x"].append(t)
                option_indices["y"].append(metaData["z_argmax"].numpy())
                t += step_count

                # Update the render
                if self.renderCriteria:
                    img = env.render()
                    self.recorded_frames.append(img)

                if done:
                    if self.gridCriteria:
                        # final agent pos
                        self.get_agent_pos(env)

                        self.plotter.plotPath(
                            self.grid,
                            self.path,
                            dir=self.gridDir,
                            epoch=str(epoch),
                        )
                        self.path = []

                    if self.renderCriteria:
                        width = self.recorded_frames[0].shape[0]
                        height = self.recorded_frames[0].shape[1]
                        self.plotter.plotRendering(
                            self.recorded_frames,
                            dir=self.renderDir,
                            epoch=str(epoch),
                            width=width,
                            height=height,
                            fps=self.render_fps,
                        )
                        self.recorded_frames = []

                        # save option indices
                        self.plotter.plotOptionIndices(
                            option_indices, dir=self.plotter.oc_path, epoch=epoch
                        )

                    dist, ep_entropy = compute_categorical_entropy(
                        option_indices["y"], policy._a_dim
                    )

                    ep_buffer.append(
                        {
                            "ep_reward": ep_reward,
                            "ep_length": ep_length,
                            "ep_entropy": ep_entropy,
                        }
                    )

        reward_list = [ep_info["ep_reward"] for ep_info in ep_buffer]
        length_list = [ep_info["ep_length"] for ep_info in ep_buffer]
        entropy_list = [ep_info["ep_entropy"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(reward_list), np.std(reward_list)
        ln_mean, ln_std = np.mean(length_list), np.std(length_list)
        ent_mean, ent_std = np.mean(entropy_list), np.std(entropy_list)
        winRate_mean, winRate_std = np.mean(successes), np.std(successes)

        eval_dict = {
            "rew_mean": rew_mean,
            "rew_std": rew_std,
            "ln_mean": ln_mean,
            "ln_std": ln_std,
            "ent_mean": ent_mean,
            "ent_std": ent_std,
            "winRate_mean": winRate_mean,
            "winRate_std": winRate_std,
        }

        if queue is not None:
            queue.put([eval_dict])
        else:
            return eval_dict

    def update_render_criteria(self, epoch, num_episodes):
        basisCriteria = epoch % self.log_interval == 0 and num_episodes == 0
        self.gridCriteria = basisCriteria and self.gridPlot
        self.renderCriteria = basisCriteria and self.renderPlot

    def init_grid(self, env):
        self.grid = np.copy(env.render()).astype(np.float32) / 255.0

    def get_agent_pos(self, env):
        # Update the grid
        if self.gridCriteria:
            self.path.append(env.get_agent_pos())
