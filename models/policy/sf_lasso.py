import time
import os
import cv2
import pickle
import numpy as np
from math import floor
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from models.layers import MLP, ConvNetwork
from models.policy.base_policy import BasePolicy


def compare_network_weights(model1: nn.Module, model2: nn.Module) -> float:
    """
    Compare the weights of two models and return the mean squared error between them.

    Args:
        model1 (nn.Module): The first model to compare.
        model2 (nn.Module): The second model to compare.

    Returns:
        float: The mean squared error between the weights of the two models.
    """
    mse_loss = nn.MSELoss()
    total_mse = 0.0
    num_params = 0

    # Iterate through parameters of both models
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if param1.shape != param2.shape:
            raise ValueError(
                "Model parameters have different shapes, models might have different architectures."
            )

        # Calculate MSE between parameters
        mse = mse_loss(param1, param2)
        total_mse += mse.item()
        num_params += 1

    # Average MSE across all parameters
    average_mse = total_mse / num_params if num_params > 0 else 0.0
    print(average_mse)

    return average_mse


class SF_LASSO(BasePolicy):
    def __init__(
        self,
        env_name: str,
        feaNet: ConvNetwork,
        feature_weights: np.ndarray,
        a_dim: int,
        sf_dim: int,
        snac_split_ratio: float,
        sf_lr: float = 1e-4,
        batch_size: int = 1024,
        reward_loss_scaler: float = 1.0,
        state_loss_scaler: float = 0.1,
        weight_loss_scaler: float = 1e-6,
        lasso_loss_scaler: float = 1.0,
        is_discrete: bool = False,
        sf_path: str | None = None,
        device: str = "cpu",
    ):
        super(SF_LASSO, self).__init__()

        ### constants
        self.sf_dim = sf_dim
        self.num_r_features = floor(sf_dim * snac_split_ratio)
        self.num_s_features = sf_dim - self.num_r_features

        self.env_name = env_name
        self.device = device
        self.batch_size = batch_size

        self._a_dim = a_dim

        self._reward_loss_scaler = reward_loss_scaler
        self._state_loss_scaler = state_loss_scaler
        self._weight_loss_scaler = weight_loss_scaler
        self._lasso_loss_scaler = lasso_loss_scaler

        self._is_discrete = is_discrete
        self._forward_steps = 0

        ### trainable networks
        self.feaNet = feaNet

        if sf_path is None:
            sf_path = "phi_prediction"
            if not os.path.exists(sf_path):
                os.mkdir(sf_path)
        self.sf_path = sf_path

        ### Define feature_weights
        self.feature_weights = nn.Parameter(
            torch.tensor(feature_weights).to(dtype=self._dtype, device=self.device),
            requires_grad=True,
        )

        # Normalize to have L2 norm = 1
        self.feature_weights.data = (
            self.feature_weights.data
            / self.feature_weights.data.norm(p=2, dim=-1, keepdim=True)
        )

        ### Define optimizers
        self.feature_optims = torch.optim.Adam(
            [{"params": self.feaNet.parameters(), "lr": sf_lr}]
        )

        #
        self.dummy = torch.tensor(1e-5)
        self.to(self.device).to(self._dtype)

    def to_device(self, device):
        self.device = device
        self.to(device)

    def preprocess_obs(self, obs):
        observation = obs["observation"]

        if not torch.is_tensor(observation):
            observation = torch.from_numpy(observation).to(self._dtype).to(self.device)

        return {"observation": observation}

    def forward(self, obs, z=None, deterministic: bool | None = False):
        self._forward_steps += 1
        obs = self.preprocess_obs(obs)

        if self._is_discrete:
            a = torch.randint(0, self._a_dim, (1,))
            a = F.one_hot(a, num_classes=self._a_dim)
        else:
            a = torch.rand((self._a_dim,))

        return a, {
            # some dummy variables to keep the code consistent across algs
            "z": self.dummy,  # dummy
            "probs": self.dummy,  # dummy
            "logprobs": self.dummy,  # dummy
            "entropy": self.dummy,  # dummy
        }

    def random_walk(self, obs):
        return self(obs)

    def get_features(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
        to_numpy: bool = False,
    ):
        phi, _ = self.feaNet(states, deterministic=deterministic)
        if to_numpy:
            phi = phi.cpu().numpy()

        return phi

    def phi_Loss(self, states, actions, next_states, rewards):
        """
        Training target: phi_r (reward), phi_s (state)  -->  (Critic: feaNet)
        Method: reward mse (r - phi_r * w), state_pred mse (s' - D(phi_s, a))
        ---------------------------------------------------------------------------
        phi ~ [N, F/2]
        w ~ [1, F/2]
        """
        phi = self.get_features(states)
        if self.num_r_features == 0:
            reward_loss = self.dummy

            state_pred = self.decode(phi, actions)
            state_loss = self._state_loss_scaler * self.mse_loss(
                state_pred, next_states
            )

            phi_r_norm = self.dummy
            phi_s_norm = torch.norm(phi, p=1)

            lasso_loss = self.dummy
            lasso_penalty = self.dummy

        elif self.num_s_features == 0:
            reward_pred = self.multiply_weights(phi, self.feature_weights)
            reward_loss = self._reward_loss_scaler * self.mse_loss(reward_pred, rewards)

            state_loss = self.dummy

            r_dim = torch.tensor(phi.shape[-1], device=self.device)

            phi_r_norm = torch.norm(phi, p=1)
            phi_s_norm = self.dummy

            lasso_loss = self._lasso_loss_scaler * phi_r_norm
            lasso_penalty = torch.relu(1e-3 * torch.sqrt(r_dim) - phi_r_norm)
        else:
            phi_r, phi_s = self.split(phi, self.num_r_features)

            reward_pred = self.multiply_weights(phi_r, self.feature_weights)
            reward_loss = self._reward_loss_scaler * self.mse_loss(reward_pred, rewards)

            state_pred = self.decode(phi_s, actions)
            state_loss = self._state_loss_scaler * self.mse_loss(
                state_pred, next_states
            )

            r_dim = torch.tensor(phi_r.shape[-1], device=self.device)

            phi_r_norm = torch.norm(phi_r, p=1)
            phi_s_norm = torch.norm(phi_s, p=1)

            lasso_loss = self._lasso_loss_scaler * phi_r_norm
            lasso_penalty = torch.relu(1e-3 * torch.sqrt(r_dim) - phi_r_norm)

        weight_norm = 0
        for param in self.feaNet.parameters():
            if param.requires_grad:  # Only include parameters that require gradients
                weight_norm += torch.norm(param, p=2)  # L

        weight_loss = self._weight_loss_scaler * weight_norm

        phi_loss = reward_loss + state_loss + weight_loss + lasso_loss + lasso_penalty

        # Plot predicted vs true rewards
        if self._forward_steps % 10 == 0:
            self.plot_rewards(reward_pred, rewards)

        return phi_loss, {
            "reward_loss": reward_loss,
            "state_loss": state_loss,
            "weight_loss": weight_loss,
            "lasso_loss": lasso_loss,
            "phi_r_norm": phi_r_norm,
            "phi_s_norm": phi_s_norm,
        }

    def decode(self, phi, actions):
        # Does some dimensional and np <-> tensor work
        # and pass it to feature decoder actions should be one-hot
        if isinstance(phi, np.ndarray):
            phi = torch.from_numpy(phi).to(self.device).to(self._dtype)
        if len(phi.shape) == 1:
            phi = phi.unsqueeze(0)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device).to(self._dtype)
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)

        reconstructed_state = self.feaNet.decode(phi, actions)
        return reconstructed_state

    def evaluate(self, buffer):
        ### Pull data from the batch
        batch = buffer.sample(self.batch_size)
        log_num = 10

        states = (
            torch.from_numpy(batch["states"]).to(self._dtype).to(self.device)[:log_num]
        )
        actions = (
            torch.from_numpy(batch["actions"]).to(self._dtype).to(self.device)[:log_num]
        )
        next_states = batch["next_states"][:log_num]
        rewards = batch["rewards"][:log_num]

        with torch.no_grad():
            phi = self.get_features(states, deterministic=True)
            reward_feature, state_feature = self.split(phi, self.num_r_features)
            reward_preds = self.multiply_weights(reward_feature, self.feature_weights)

        if self.num_s_features != 0:
            ### decoder reconstructed images ###
            ground_truth_images = []
            predicted_images = []
            for i in range(phi.shape[0]):
                # Decode the feature and append to reconstructed states
                with torch.no_grad():
                    true_state = next_states[i]
                    decoded_state = self.decode(state_feature[i], actions[i])
                    decoded_state = decoded_state.squeeze(0)

                # Handle vector data by reshaping into a heatmap

                if decoded_state.dim() == 1:  # If the state is a vector
                    heatmap_data = (
                        decoded_state.cpu().numpy().reshape(1, -1)
                    )  # Reshape for heatmap
                else:  # Assume it's already 2D
                    heatmap_data = decoded_state.cpu().numpy()

                # Normalize the heatmap data between 0 and 1
                true_image = (true_state - np.min(true_state)) / (
                    np.max(true_state) - np.min(true_state)
                )
                pred_image = (heatmap_data - np.min(heatmap_data)) / (
                    np.max(heatmap_data) - np.min(heatmap_data)
                )

                # Update the corresponding subplot
                true_image, pred_image = self.get_image(true_image, pred_image)
                ground_truth_images.append(true_image)
                predicted_images.append(pred_image)
        else:
            ground_truth_images = [None]
            predicted_images = [None]

        if self.num_r_features != 0:
            ### create reward plot ###
            x = range(log_num)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.stem(
                x,
                rewards,
                linefmt="r-",
                markerfmt="ro",
                basefmt="k-",
                label="True Rewards",
            )
            ax.stem(
                x,
                reward_preds.cpu().numpy(),
                linefmt="b-",
                markerfmt="bo",
                basefmt="k-",
                label="Predicted Rewards",
            )

            # Set logarithmic y-scale
            # ax.set_yscale('log')
            ax.set_xlabel("Reward Index")
            ax.set_ylabel("Reward")
            ax.set_title("Predicted vs True Rewards")
            ax.legend()
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            plt.tight_layout()

            # Render the figure to a canvas
            canvas = FigureCanvas(fig)
            canvas.draw()

            # Convert canvas to a NumPy array
            reward_pred_img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
            reward_pred_img = reward_pred_img.reshape(
                canvas.get_width_height()[::-1] + (3,)
            )  # Shape: (height, width, 3)
            plt.close()
        else:
            reward_pred_img = None

        ### FEATURE IMAGE ###
        phi = phi.cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(
            phi,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title("Feature heatmap")
        plt.tight_layout()

        # Render the figure to a canvas
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Convert canvas to a NumPy array
        feature_img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        feature_img = feature_img.reshape(
            canvas.get_width_height()[::-1] + (3,)
        )  # Shape: (height, width, 3)
        plt.close()

        return {
            "ground_truth": ground_truth_images,
            "prediction": predicted_images,
            "reward_plot": [reward_pred_img],
            "feature_plot": [feature_img],
        }

    def learn(self, buffer):
        self.train()
        t0 = time.time()

        ### Pull data from the batch
        batch = buffer.sample(self.batch_size)

        states = torch.from_numpy(batch["states"]).to(self._dtype).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self._dtype).to(self.device)
        next_states = (
            torch.from_numpy(batch["next_states"]).to(self._dtype).to(self.device)
        )
        rewards = torch.from_numpy(batch["rewards"]).to(self._dtype).to(self.device)

        ### Update
        phi_loss, phi_loss_dict = self.phi_Loss(states, actions, next_states, rewards)

        self.feature_optims.zero_grad()
        phi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        phi_grad_dict = self.compute_gradient_norm(
            [self.feaNet],
            ["feaNet"],
            dir="SF",
            device=self.device,
        )
        norm_dict = self.compute_weight_norm(
            [self.feaNet],
            ["feaNet"],
            dir="SF",
            device=self.device,
        )
        self.feature_optims.step()

        ### Logging
        loss_dict = {
            "SF/loss": phi_loss.item(),
            "SF/reward_loss": phi_loss_dict["reward_loss"].item(),
            "SF/state_loss": phi_loss_dict["state_loss"].item(),
            "SF/weight_loss": phi_loss_dict["weight_loss"].item(),
            # "SF/orthogonal_loss": phi_loss_dict["orthogonal_loss"].item(),
            "SF/lasso_loss": phi_loss_dict["lasso_loss"].item(),
            "SF/phi_r_norm": phi_loss_dict["phi_r_norm"].item(),
            "SF/phi_s_norm": phi_loss_dict["phi_s_norm"].item(),
        }
        loss_dict.update(norm_dict)
        loss_dict.update(phi_grad_dict)

        t1 = time.time()
        self.eval()
        return loss_dict, t1 - t0

    def get_image(self, true_image, pred_image):
        is_image = True if len(true_image.shape) == 3 else False

        if is_image:
            img_list = []
            for img in [true_image, pred_image]:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(img)
                plt.tight_layout()
                # Render the figure to a canvas
                canvas = FigureCanvas(fig)
                canvas.draw()
                # Convert canvas to a NumPy array
                image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
                image = image.reshape(
                    canvas.get_width_height()[::-1] + (3,)
                )  # Shape: (height, width, 3)
                img_list.append(image)
                plt.close()
            true_image = img_list[0]
            pred_image = img_list[1]
        else:
            true_image = true_image.flatten()
            pred_image = pred_image.flatten()

            # Plot stem
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(true_image))
            ax.stem(
                x,
                true_image,
                linefmt="r-",
                markerfmt="ro",
                basefmt="k-",
                label="True states",
            )
            ax.stem(
                x,
                pred_image,
                linefmt="b-",
                markerfmt="bo",
                basefmt="k-",
                label="Predicted States",
            )

            # Set logarithmic y-scale
            plt.title("True vs Predicted States")
            plt.legend()
            plt.tight_layout()
            # Render the figure to a canvas
            canvas = FigureCanvas(fig)
            canvas.draw()
            # Convert canvas to a NumPy array
            image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
            true_image = image.reshape(
                canvas.get_width_height()[::-1] + (3,)
            )  # Shape: (height, width, 3)
            pred_image = None
            plt.close()

        return true_image, pred_image

    def plot_rewards(self, reward_pred, rewards):
        """
        Plot predicted and true rewards as a stem plot with logarithmic y-axis.
        """
        # Detach tensors and convert to NumPy for plotting
        reward_pred_np = reward_pred.detach().cpu().numpy().flatten()
        rewards_np = rewards.detach().cpu().numpy().flatten()

        # Plot stem
        x = range(len(rewards_np))
        plt.figure(figsize=(12, 6))
        plt.stem(
            x,
            rewards_np,
            linefmt="r-",
            markerfmt="ro",
            basefmt="k-",
            label="True Rewards",
        )
        plt.stem(
            x,
            reward_pred_np,
            linefmt="b-",
            markerfmt="bo",
            basefmt="k-",
            label="Predicted Rewards",
        )

        # Set logarithmic y-scale
        # plt.yscale('log')
        plt.xlabel("Reward Index")
        plt.ylabel("Reward")
        plt.title("Predicted vs True Rewards")
        plt.legend()
        plt.grid(True, which="both", ls="--", linewidth=0.5)
        plt.savefig(f"{self.sf_path}/{self._forward_steps}_reward.png")
        plt.close()

    def save_model(self, logdir, epoch=None, is_best=False):
        self.feaNet = self.feaNet.cpu()
        feature_weights = self.feature_weights.detach().clone().cpu()

        # save checkpoint
        if is_best:
            path = os.path.join(logdir, "best_model.p")
        else:
            path = os.path.join(logdir, "model_" + str(epoch) + ".p")
        pickle.dump(
            (self.feaNet, feature_weights),
            open(path, "wb"),
        )

        self.feaNet = self.feaNet.to(self.device)
