import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy

from utils.plotter import Plotter
from log.wandb_logger import WandbLogger
from models.evaulators.base_evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter


from PIL import Image, ImageDraw, ImageFont


def add_number_to_raw_image(
    image: np.ndarray,
    number: int,
    position=(10, 30),
    font_scale=1,
    color=(0, 0, 255),
    thickness=2,
):
    """
    Adds a number to the top-left corner of a raw NumPy array image.

    Args:
    - image: A NumPy array representing the image.
    - number: The number to add to the image.
    - position: Tuple (x, y) for the text position in pixels.
    - font_scale: Scale of the text.
    - color: Color of the text in BGR (e.g., (0, 0, 255) for red).
    - thickness: Thickness of the text.

    Returns:
    - The modified NumPy array image with the number added.
    """
    # Make a copy of the image to avoid modifying the original
    annotated_image = image.copy()

    # Add the number to the image using OpenCV's putText function
    cv2.putText(
        annotated_image,
        str(number),  # The text to add
        position,  # Position of the text
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        fontScale=font_scale,  # Font size
        color=color,  # Text color in BGR format
        thickness=thickness,  # Text thickness
        lineType=cv2.LINE_AA,  # Line type for smooth text
    )
    return annotated_image


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
    indices_flat = np.concatenate(indices)

    # Count occurrences of each category
    counts = np.bincount(indices_flat, minlength=num_categories)

    # Normalize to get the categorical distribution
    distribution = counts / counts.sum()

    # Compute the entropy
    entropy_value = entropy(distribution, base=2)  # Use base-2 for bits

    return distribution, entropy_value


class HC_Evaluator(Evaluator):
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
        min_option_length: int = 3,
        gamma: float = 0.99,
        eval_ep_num: int = 1,
        episode_len: int = 100,
        log_interval: int = 1,
    ):
        super(HC_Evaluator, self).__init__(
            logger=logger,
            writer=writer,
            training_env=training_env,
            testing_env=testing_env,
            eval_ep_num=eval_ep_num,
            log_interval=log_interval,
        )
        self.plotter = plotter
        self.render_fps = render_fps
        self.gamma = gamma
        self.min_option_length = min_option_length
        self.episode_len = episode_len

        if dir is not None:
            if gridPlot:
                self.gridPlot = True
                self.gridDir = os.path.join(dir, "grid")
                if not os.path.exists(self.gridDir):
                    os.mkdir(self.gridDir)
                self.path = []
                self.path_marker = []
            else:
                self.gridPlot = False
            if renderPlot:
                self.renderPlot = True
                self.renderDir = os.path.join(dir, "render")
                if not os.path.exists(self.renderDir):
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
    ) -> dict[str, list[float]]:
        ep_buffer = []

        # For each episode, apply different seed for stochasticity
        if seed is None:
            seed = random.randint(0, 1_000_000)

        if queue is not None:
            self.set_any_seed(grid_type, seed)

        def env_step(a, is_option: bool = False):
            next_obs, rew, term, trunc1, infos = env.step(a)

            if self.gridCriteria:
                self.get_agent_pos(env, is_option=is_option)
            if self.renderCriteria:
                img = env.render()
                img = add_number_to_raw_image(
                    img,
                    number=self.external_t,
                    position=(10, 30),
                    font_scale=1,
                    color=(255, 0, 0),
                    thickness=2,
                )
                self.recorded_frames.append(img)

            self.external_t += 1
            trunc2 = True if self.external_t == self.episode_len else False
            done = term or trunc1 or trunc2

            return next_obs, rew, done, infos

        successes = np.zeros((self.eval_ep_num,))
        failures = np.zeros((self.eval_ep_num,))
        for num_episodes in range(self.eval_ep_num):
            self.update_render_criteria(epoch, num_episodes)

            # logging initialization
            ep_reward = 0

            # env initialization
            options = {"random_init_pos": False}
            obs, _ = env.reset(seed=grid_type, options=options)

            if self.gridCriteria:
                self.init_grid(env)
                self.get_agent_pos(env)

            option_indices = {"x": [], "y": []}
            done = False
            self.external_t = 1
            while not done:
                with torch.no_grad():
                    a, metaData = policy(obs, idx, deterministic=False)
                    a = a.cpu().numpy().squeeze() if a.shape[-1] > 1 else [a.item()]

                option_indices["x"].append(self.external_t)
                option_indices["y"].append(metaData["z_argmax"].numpy())

                ### Create an Option Loop
                if metaData["is_option"]:
                    next_obs, rew, done, infos = env_step(
                        a, is_option=metaData["is_option"]
                    )
                    if not done:
                        for o_t in range(1, self.min_option_length):
                            # env stepping
                            with torch.no_grad():
                                option_a, _ = policy(
                                    next_obs,
                                    metaData["z_argmax"],
                                    deterministic=False,
                                )
                                option_a = option_a.cpu().numpy().squeeze()

                            next_obs, op_rew, done, infos = env_step(option_a)
                            rew += self.gamma**o_t * op_rew
                            if done:
                                break

                else:
                    ### Conventional Loop
                    next_obs, rew, done, infos = env_step(a)

                obs = next_obs

                if "success" in infos:
                    successes[num_episodes] = np.maximum(
                        successes[num_episodes], infos["success"]
                    )
                if "failures" in infos:
                    failures[num_episodes] = np.maximum(
                        failures[num_episodes], infos["failures"]
                    )

                ep_reward += rew

                if done:
                    dist, ep_entropy = compute_categorical_entropy(
                        option_indices["y"], policy._a_dim
                    )

                    ep_buffer.append(
                        {
                            "ep_reward": ep_reward,
                            "ep_length": self.external_t,
                            "ep_entropy": ep_entropy,
                        }
                    )

                    if self.gridCriteria:
                        # final agent pos
                        self.get_agent_pos(env)

                        self.plotter.plotPath(
                            self.grid,
                            self.path,
                            dir=self.gridDir,
                            epoch=str(epoch),
                            path_marker=self.path_marker,
                        )
                        self.path = []
                        self.path_marker = []

                        # save option indices
                        self.plotter.plotOptionIndices(
                            option_indices, dir=self.plotter.hc_path, epoch=epoch
                        )

                    if self.renderCriteria:
                        # save rendering
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

        reward_list = [ep_info["ep_reward"] for ep_info in ep_buffer]
        length_list = [ep_info["ep_length"] for ep_info in ep_buffer]
        entropy_list = [ep_info["ep_entropy"] for ep_info in ep_buffer]

        rew_mean, rew_std = np.mean(reward_list), np.std(reward_list)
        ln_mean, ln_std = np.mean(length_list), np.std(length_list)
        ent_mean, ent_std = np.mean(entropy_list), np.std(entropy_list)
        winRate_mean, winRate_std = np.mean(successes), np.std(successes)
        failRate_mean, failRate_std = np.mean(failures), np.std(failures)

        eval_dict = {
            "rew_mean": rew_mean,
            "rew_std": rew_std,
            "ln_mean": ln_mean,
            "ln_std": ln_std,
            "ent_mean": ent_mean,
            "ent_std": ent_std,
            "winRate_mean": winRate_mean,
            "winRate_std": winRate_std,
            "failRate_mean": failRate_mean,
            "failRate_std": failRate_std,
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

    def get_agent_pos(self, env, is_option: bool = False):
        # Update the grid
        if self.gridCriteria:
            self.path.append(env.get_agent_pos())
            self.path_marker.append(is_option)
