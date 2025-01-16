import cv2
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.plotter import Plotter
from log.wandb_logger import WandbLogger
from models.evaulators.base_evaluator import Evaluator
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


class UG_Evaluator(Evaluator):
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env,
        plotter: Plotter,
        testing_env=None,
        dir: str = None,
        gridPlot: bool = True,
        renderPlot: bool = True,
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

        if dir is not None:
            if gridPlot:
                self.gridPlot = True
                self.gridDir = os.path.join(dir, "grid")
                os.mkdir(self.gridDir)
                self.paths = []
            else:
                self.gridPlot = False
            if renderPlot:
                self.renderPlot = True
                self.renderDir = os.path.join(dir, "render")
                os.mkdir(self.renderDir)
                self.uniform_dir = os.path.join(dir, "render", "uniform")
                os.mkdir(self.uniform_dir)
                self.greedy_dir = os.path.join(dir, "render", "greedy")
                os.mkdir(self.greedy_dir)
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
        env_seed: int = 0,
        seed: int = None,
        queue=None,
    ) -> Dict[str, List[float]]:
        ep_buffer = []
        if queue is not None:
            self.set_any_seed(env_seed, seed)

        for index, actor in enumerate(policy):
            path = []
            self.update_render_criteria(epoch, 0)

            # logging initialization
            ep_reward, ep_length = 0, 0

            # env initialization
            options = {"random_init_pos": False}
            s, _ = env.reset(seed=env_seed, options=options)

            if self.gridCriteria:
                self.init_grid(env)

            done = False
            while not done:
                with torch.no_grad():
                    a, phi_dict = actor(s, idx, deterministic=True)
                    a = a.cpu().numpy().squeeze()

                ns, rew, term, trunc, _ = env.step(a)
                done = term or trunc

                s = ns
                ep_reward += rew
                ep_length += 1

                if self.gridCriteria:
                    if hasattr(env.env, "agent_pos"):
                        self.path.append(env.get_wrapper_attr("agent_pos"))
                    elif hasattr(env.env, "agents"):
                        self.path.append(env.get_wrapper_attr("agents")[0].pos)
                    else:
                        raise ValueError("No agent position information.")

                # Update the render
                if self.renderCriteria:
                    img = env.render()
                    self.recorded_frames.append(img)

                if done:
                    if self.gridCriteria:
                        self.paths.append(path)

                    if self.renderCriteria:
                        video_path = self.uniform_dir if index == 0 else self.greedy_dir
                        width = self.recorded_frames[0].shape[0]
                        height = self.recorded_frames[0].shape[1]
                        self.plotter.plotRendering(
                            self.recorded_frames,
                            dir=video_path,
                            epoch=idx,
                            width=width,
                            height=height,
                        )
                        self.recorded_frames = []

                    ep_buffer.append({"reward": ep_reward, "ep_length": ep_length})

        if self.gridCriteria:
            self.plotter.plotPath2(
                self.grid,
                self.paths,
                # dir=os.path.join(self.gridDir, str(epoch)),
                dir=self.gridDir,
                epoch=idx,
            )
            self.paths = []

        reward_list = [ep_info["reward"] for ep_info in ep_buffer]
        length_list = [ep_info["ep_length"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(reward_list), np.std(reward_list)
        ln_mean, ln_std = np.mean(length_list), np.std(length_list)

        if queue is not None:
            queue.put([rew_mean, rew_std, ln_mean, ln_std])
        else:
            return rew_mean, rew_std, ln_mean, ln_std

    def update_render_criteria(self, epoch, num_episodes):
        basisCriteria = epoch % self.log_interval == 0 and num_episodes == 0
        self.gridCriteria = basisCriteria and self.gridPlot
        self.renderCriteria = basisCriteria and self.renderPlot

    def init_grid(self, env):
        self.grid = np.copy(env.render()).astype(np.float32) / 255.0
