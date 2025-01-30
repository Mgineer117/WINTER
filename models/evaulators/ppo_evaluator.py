import cv2
import os
import torch
import gymnasium as gym
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from utils.plotter import Plotter
from log.wandb_logger import WandbLogger
from models.evaulators.base_evaluator import Evaluator
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


class PPO_Evaluator(Evaluator):
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
        eval_ep_num: int = 10,
    ):
        super().__init__(
            logger=logger,
            writer=writer,
            training_env=training_env,
            testing_env=testing_env,
            eval_ep_num=eval_ep_num,
        )
        self.plotter = plotter

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
        idx: int = None,
        grid_type: int = 0,
    ) -> Dict[str, List[float]]:

        ep_buffer = []

        successes = np.zeros((self.eval_ep_num,))
        for num_episodes in range(self.eval_ep_num):
            # logging initialization
            ep_reward, ep_length = 0, 0

            # env initialization
            options = {"random_init_pos": False}
            s, _ = env.reset(seed=grid_type, options=options)

            if num_episodes == 0 and self.gridPlot:
                self.init_grid(env)

            done = False
            while not done:
                with torch.no_grad():
                    a, _ = policy(s, idx, deterministic=True)
                    a = a.cpu().numpy().squeeze() if a.shape[-1] > 1 else [a.item()]

                ns, rew, term, trunc, infos = env.step(a)
                done = term or trunc

                s = ns
                if "success" in infos:
                    successes[num_episodes] = np.maximum(
                        successes[num_episodes], infos["success"]
                    )
                ep_reward += rew
                ep_length += 1

                # Update the grid
                if num_episodes == 0 and self.gridPlot:
                    self.get_agent_pos(env)
                # Update the render
                if num_episodes == 0 and self.renderPlot:
                    img = env.render()
                    self.recorded_frames.append(img)
                
                if done:
                    if num_episodes == 0 and self.gridPlot:
                        # final agent pos
                        self.get_agent_pos(env)

                        path_image = self.plotPath()
                        self.path = []

                    if num_episodes == 0 and self.renderPlot:
                        path_render = self.plotRender()
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

        supp_dict = {"path_image": path_image, "path_render": path_render}

        return eval_dict, supp_dict

    def plotPath(self):
        grid = self.grid
        path = self.path
        path_marker = None
        img_tile_size = 32

        # Rotate the grid and set up the figure
        grid = np.rot90(grid, k=1)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, origin="upper")
        ax.axis("off")

        img_size = grid.shape[0]
        path_length = len(path) - 1

        for idx, (i_point, f_point) in enumerate(zip(path[:-1], path[1:])):
            # Calculate coordinates
            y = [(p[0] * img_tile_size + img_tile_size / 2) for p in [i_point, f_point]]
            x = [(p[1] * img_tile_size + img_tile_size / 2) for p in [i_point, f_point]]
            y = [img_size - yi for yi in y]

            # Plot path components
            if idx == 0:
                ax.scatter(x[0], y[0], color="red", s=30)  # Start point
            if idx == path_length:
                ax.scatter(x[1], y[1], color="blue", s=30)  # End point
            if path_marker and path_marker[idx]:
                ax.scatter(
                    x[0],
                    y[0],
                    color="yellow",
                    marker="*",
                    s=100,
                    edgecolors="black",
                    linewidths=1.0,
                )
            if (i_point != f_point).any():
                ax.plot(x, y, color="green", linewidth=2)  # Path line

        # Convert figure to a NumPy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(
            canvas.get_width_height()[::-1] + (3,)
        )  # Shape: (height, width, 3)

        plt.close(fig)
        return img

    def plotRender(self):
        # Assuming self.recorded_frames is a list of frames, where each frame is a 2D or 3D array
        frames = []

        for frame in self.recorded_frames:
            # Ensure the frame is a NumPy array
            frame = np.array(frame)

            # Handle grayscale frames (2D arrays), convert to RGB
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=2)  # Convert to (H, W, 3)

            frames.append(frame)

        # Stack all frames into a single NumPy array (N, C, H, W)
        frames = np.stack(frames, axis=0)

        # Example: You can now use frames with wandb.Video
        return frames

    def init_grid(self, env):
        self.grid = np.copy(env.render()).astype(np.float32) / 255.0

    def get_agent_pos(self, env):
        # Update the grid
        self.path.append(env.get_agent_pos())
