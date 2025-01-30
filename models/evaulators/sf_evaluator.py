import cv2
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from copy import deepcopy

from utils.plotter import Plotter
from utils.utils import generate_2d_heatmap_image
from log.wandb_logger import WandbLogger
from log.logger_util import RunningAverage, colorize, convert_json
from models.evaulators.base_evaluator import Evaluator
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


class SF_Evaluator(Evaluator):
    def __init__(
        self,
        logger: WandbLogger,
        writer: SummaryWriter,
        training_env,
        plotter: Plotter,
        testing_env: None = None,
        dir: str = None,
        featurePlot: bool = False,
        eval_ep_num: int = 1,
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
            if featurePlot:
                ### feature rendering
                self.featurePlot = True
                self.featureDir = os.path.join(dir, "feature")
                if os.path.exists(self.featureDir):
                    warning_msg = colorize(
                        "Warning: Log dir %s already exists! Some logs may be overwritten."
                        % self.featureDir,
                        "magenta",
                        True,
                    )
                    print(warning_msg)
                else:
                    os.mkdir(self.featureDir)
                self.feature_frames = []
            else:
                self.featurePlot = False
        else:
            self.featurePlot = False

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

            if self.eigenCriteria:
                self.init_grid(env)

            done = False
            while not done:
                with torch.no_grad():
                    a, phi_dict = policy(obs, idx, deterministic=False)
                    a = a.cpu().numpy().squeeze()

                # env stepping
                next_obs, rew, term, trunc, infos = env.step(a)
                ns = next_obs["observation"]

                done = term or trunc

                # Update the render
                if self.featureCriteria:
                    phi_s = phi_dict["phi_s"]
                    phi_r = phi_dict["phi_r"]

                    state_pred = policy.feaNet.decode(
                        phi_s, phi_dict["a_oh"], phi_dict["conv_dict"]
                    )

                    img1 = (
                        state_pred.squeeze(0).clone().detach().cpu().numpy()
                    )  # [:, :, 1]
                    img2 = ns  # [:, :, 1]

                    # image is 2d (n, n, 1); so expanding
                    img1 = np.repeat(img1, 3, axis=2)
                    img2 = np.repeat(img2, 3, axis=2)

                    img1 = np.repeat(
                        np.repeat(img1, self.plotter.img_tile_size, axis=0),
                        self.plotter.img_tile_size,
                        axis=1,
                    )
                    img2 = np.repeat(
                        np.repeat(img2, self.plotter.img_tile_size, axis=0),
                        self.plotter.img_tile_size,
                        axis=1,
                    )

                    s_img = np.hstack((img1, img2)) * 50.0
                    r_img = generate_2d_heatmap_image(
                        phi_r.squeeze(0).clone().detach().cpu().numpy(),
                        img_size=s_img.shape[0],
                    )

                    combined_img = np.hstack((r_img, s_img)).astype(np.uint8)

                    self.feature_frames.append(combined_img)

                if self.renderCriteria:
                    img = env.render()
                    self.render_frames.append(img)

                obs = next_obs

                if "success" in infos:
                    successes[num_episodes] = np.maximum(
                        successes[num_episodes], infos["success"]
                    )
                ep_reward += rew
                ep_length += 1

                if done:
                    if self.eigenCriteria:
                        self.plotter.plotEigenFunction1(
                            eigenvectors=policy._options.clone().detach().numpy().T,
                            dir=self.eigenDir,
                            epoch=epoch,
                        )

                    if self.gridCriteria:
                        self.plotter.plotFeature(
                            self.grid,
                            self.grid_r,
                            self.grid_s,
                            self.grid_v,
                            self.grid_q,
                            dir=self.gridDir,
                            epoch=epoch,
                        )

                    if self.renderCriteria:
                        width = self.render_frames[0].shape[0]
                        height = self.render_frames[0].shape[1]
                        self.plotter.plotRendering(
                            self.render_frames,
                            dir=self.renderDir,
                            epoch=epoch,
                            width=width,
                            height=height,
                        )
                        self.render_frames = []

                    if self.featureCriteria:
                        # feature rendering
                        width = self.feature_frames[0].shape[0]
                        height = self.feature_frames[0].shape[1]
                        self.plotter.plotRendering(
                            self.feature_frames,
                            dir=self.featureDir,
                            epoch=epoch,
                            width=width,
                            height=height,
                            fps=self.render_fps,
                        )
                        self.feature_frames = []

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
        self.eigenCriteria = basisCriteria and self.eigenPlot
        self.gridCriteria = basisCriteria and self.gridPlot
        self.renderCriteria = basisCriteria and self.renderPlot
        self.featureCriteria = basisCriteria and self.featurePlot

    def init_grid(self, env):
        self.grid = np.copy(env.render()).astype(np.float32) / 255.0

        self.grid_r = np.zeros(self.grid.shape) + 1e-5
        self.grid_s = np.zeros(self.grid.shape) + 1e-5
        self.grid_v = np.zeros(self.grid.shape) + 1e-5
        self.grid_q = np.zeros(self.grid.shape) + 1e-5

    def update_grid(self, coord, phi_r, phi_s, q):
        ### create image
        phi_r, phi_s = phi_r.cpu().numpy(), phi_s.cpu().numpy()

        phi_r = np.sum(phi_r) / phi_r.shape[-1]
        phi_s = np.sum(phi_s) / phi_s.shape[-1]

        colormap = cm.get_cmap("gray")  # Choose a colormap
        color_r = colormap(phi_r * 5)[:3]  # Get RGB values
        color_s = colormap(phi_s * 2)[:3]  # Get RGB values
        color_q = colormap(q)[:3]  # Get RGB values
        # coord =
        coordx = [
            coord[0] * self.plotter.img_tile_size + 1,
            coord[0] * self.plotter.img_tile_size + self.plotter.img_tile_size - 1,
        ]
        coordy = [
            coord[1] * self.plotter.img_tile_size + 1,
            coord[1] * self.plotter.img_tile_size + self.plotter.img_tile_size - 1,
        ]

        self.grid_r[coordx[0] : coordx[1], coordy[0] : coordy[1], :] = color_r
        self.grid_s[coordx[0] : coordx[1], coordy[0] : coordy[1], :] = color_s
        self.grid_v[coordx[0] : coordx[1], coordy[0] : coordy[1], :] += (
            0.01,
            0.01,
            0.01,
        )
        self.grid_q[coordx[0] : coordx[1], coordy[0] : coordy[1], :] = color_q

    def plot_options(self, S, V):
        self.plotter.plotEigenFunctionAll(S.numpy(), V.numpy())
        self.eigenPlot = False
