import cv2
import os
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns

from utils.plotter import Plotter
from log.wandb_logger import WandbLogger
from models.evaulators.base_evaluator import Evaluator
from torch.utils.tensorboard import SummaryWriter


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
        min_option_length: int = 3,
        gamma: float = 0.99,
        eval_ep_num: int = 10,
        episode_len: int = 100,
    ):
        super(HC_Evaluator, self).__init__(
            logger=logger,
            writer=writer,
            training_env=training_env,
            testing_env=testing_env,
            eval_ep_num=eval_ep_num,
        )
        self.plotter = plotter
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
        idx: int = None,
        grid_type: int = 0,
    ) -> dict[str, list[float]]:

        ep_buffer = []
        path_image = None  # placeholder
        path_render = None  # placeholder

        def env_step(a, is_option: bool = False):
            next_obs, rew, term, trunc1, infos = env.step(a)

            if num_episodes == 0 and self.gridPlot:
                self.get_agent_pos(env, is_option=is_option)
            if num_episodes == 0 and self.renderPlot:
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
            # logging initialization
            ep_reward = 0

            # env initialization
            options = {"random_init_pos": False}
            obs, _ = env.reset(seed=grid_type, options=options)

            if num_episodes == 0 and self.gridPlot:
                self.init_grid(env)

            done = False
            self.external_t = 1
            self.option_indices = {"x": [], "y": []}
            while not done:
                with torch.no_grad():
                    a, metaData = policy(obs, idx, deterministic=False)
                    a = a.cpu().numpy().squeeze() if a.shape[-1] > 1 else [a.item()]

                self.option_indices["x"].append(self.external_t)
                self.option_indices["y"].append(metaData["z_argmax"].numpy())

                ### Create an Option Loop
                if metaData["is_option"]:
                    next_obs, rew, done, infos = env_step(
                        a, is_option=metaData["is_option"]
                    )
                    if not done:
                        for o_t in range(1, self.min_option_length):
                            # env stepping
                            with torch.no_grad():
                                option_a, option_dict = policy(
                                    next_obs,
                                    metaData["z_argmax"],
                                    deterministic=False,
                                )
                                option_a = option_a.cpu().numpy().squeeze()

                            next_obs, op_rew, done, infos = env_step(option_a)
                            rew += self.gamma**o_t * op_rew
                            if done or option_dict["option_termination"]:
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
                        self.option_indices["y"], policy._a_dim
                    )

                    ep_buffer.append(
                        {
                            "ep_reward": ep_reward,
                            "ep_length": self.external_t,
                            "ep_entropy": ep_entropy,
                        }
                    )

                    if num_episodes == 0 and self.gridPlot:
                        # final agent pos
                        self.get_agent_pos(env)

                        path_image = self.plotPath()
                        self.path = []

                        # save option indices
                        option_image = self.plotOptionIndices()
                    else:
                        option_image = None

                    if num_episodes == 0 and self.renderPlot:
                        path_render = self.plotRender()
                        self.recorded_frames = []
                    else:
                        path_image = None
                        path_render = None

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

        supp_dict = {
            "path_image": path_image,
            "path_render": path_render,
            "option_image": option_image,
        }

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

    def plotOptionIndices(self):
        sns.set_theme()

        option_indices = self.option_indices

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(option_indices["x"], option_indices["y"])

        # Convert figure to a NumPy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(
            canvas.get_width_height()[::-1] + (3,)
        )  # Shape: (height, width, 3)

        plt.close()
        return img

    def init_grid(self, env):
        self.grid = np.copy(env.render()).astype(np.float32) / 255.0

    def get_agent_pos(self, env, is_option: bool = False):
        # Update the grid
        self.path.append(env.get_agent_pos())
        self.path_marker.append(is_option)
