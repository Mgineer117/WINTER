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


class OP_Evaluator(Evaluator):
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env,
        plotter: Plotter,
        testing_env=None,
        dir: str = None,
        gridPlot: bool = False,
        renderPlot: bool = False,
        render_fps: int = 10,
        eval_ep_num: int = 1,
        log_interval: int = 1,
    ):
        super(OP_Evaluator, self).__init__(
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
            obs, _ = env.reset(seed=grid_type, options=options)

            if self.gridCriteria:
                self.init_grid(env)

            done = False
            while not done:
                with torch.no_grad():
                    a, _ = policy(obs, idx, deterministic=False)
                    a = a.cpu().numpy().squeeze()

                # env stepping
                next_obs, rew, term, trunc, infos = env.step(a)
                done = term or trunc

                obs = next_obs

                if "success" in infos:
                    successes[num_episodes] = np.maximum(
                        successes[num_episodes], infos["success"]
                    )
                ep_reward += rew
                ep_length += 1

                # Update the render
                if self.renderCriteria:
                    img = env.render()
                    self.recorded_frames.append(img)

                if done:
                    if self.gridCriteria:
                        self.plotter.plotPath(
                            self.grid,
                            self.path,
                            dir=os.path.join(self.gridDir, str(epoch)),
                            epoch=f"{idx}_{name1:3f}",
                        )
                        self.path = []

                    if self.renderCriteria:
                        width = self.recorded_frames[0].shape[0]
                        height = self.recorded_frames[0].shape[1]
                        self.plotter.plotRendering(
                            self.recorded_frames,
                            dir=os.path.join(self.renderDir, str(epoch)),
                            epoch=idx,
                            width=width,
                            height=height,
                            fps=self.render_fps,
                        )
                        self.recorded_frames = []

                    ep_buffer.append({"reward": ep_reward, "ep_length": ep_length})

        reward_list = [ep_info["reward"] for ep_info in ep_buffer]
        length_list = [ep_info["ep_length"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(reward_list), np.std(reward_list)
        ln_mean, ln_std = np.mean(length_list), np.std(length_list)
        winRate_mean, winRate_std = np.mean(successes), np.std(successes)

        eval_dict = {
            "rew_mean": rew_mean,
            "rew_std": rew_std,
            "ln_mean": ln_mean,
            "ln_std": ln_std,
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


class OP_Evaluator2(Evaluator):
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env,
        plotter: Plotter,
        testing_env=None,
        dir: str = None,
        gridPlot: bool = True,
        renderPlot: bool = False,
        render_fps: int = 10,
        eval_ep_num: int = 1,
        log_interval: int = 1,
    ):
        super(OP_Evaluator2, self).__init__(
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

        red_flag_captured = np.zeros((self.eval_ep_num,))
        for num_episodes in range(self.eval_ep_num):
            self.update_render_criteria(epoch, num_episodes)

            # logging initialization
            ep_reward, ep_length = 0, 0

            # env initialization
            s, _ = env.reset(seed=grid_type)

            if self.gridCriteria:
                self.init_grid(env)

            done = False
            while not done:
                with torch.no_grad():
                    a, phi_dict = policy(s, idx, deterministic=True)
                    a = a.cpu().numpy().squeeze()

                # Update the grid
                ns, rew, term, trunc, infos = env.step(a)
                done = term or trunc

                s = ns

                if "red_flag_captured" in infos:
                    red_flag_captured[num_episodes] = np.maximum(
                        red_flag_captured[num_episodes], infos["red_flag_captured"]
                    )
                ep_reward += rew
                ep_length += 1

                # Update the render
                if self.renderCriteria:
                    img = env.render()
                    self.recorded_frames.append(img)

                if done:
                    if self.gridCriteria:
                        self.plotter.plotPath(
                            self.grid,
                            self.path,
                            dir=os.path.join(self.gridDir, str(idx)),
                            epoch=f"{str(epoch)}_{name1:3f}",
                        )
                        self.path = []

                    if self.renderCriteria:
                        width = self.recorded_frames[0].shape[0]
                        height = self.recorded_frames[0].shape[1]
                        self.plotter.plotRendering(
                            self.recorded_frames,
                            dir=os.path.join(self.renderDir, str(idx)),
                            epoch=idx,
                            width=width,
                            height=height,
                            fps=self.render_fps,
                        )
                        self.recorded_frames = []

                    ep_buffer.append({"reward": ep_reward, "ep_length": ep_length})

        reward_list = [ep_info["reward"] for ep_info in ep_buffer]
        length_list = [ep_info["ep_length"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(reward_list), np.std(reward_list)
        ln_mean, ln_std = np.mean(length_list), np.std(length_list)
        winRate_mean, winRate_std = np.mean(red_flag_captured), np.std(
            red_flag_captured
        )

        eval_dict = {
            "rew_mean": rew_mean,
            "rew_std": rew_std,
            "ln_mean": ln_mean,
            "ln_std": ln_std,
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
