import time
import os
import cv2
import pickle
import numpy as np
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib

from models.layers import MLP, ConvNetwork
from models.policy.base_policy import BasePolicy

matplotlib.use("Agg")


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


def generate_2d_heatmap_image(Z, img_size):
    # Create a 2D heatmap and save it as an image
    fig, ax = plt.subplots(figsize=(5, 5))

    # Example data for 2D heatmap
    vector_length = Z.shape[0]
    grid_size = int(np.sqrt(vector_length))

    if grid_size**2 != vector_length:
        raise ValueError(
            "The length of the eigenvector must be a perfect square to reshape into a grid."
        )

    Z = Z.reshape((grid_size, grid_size))

    norm_Z = np.linalg.norm(Z)
    # Plot heatmap
    heatmap = ax.imshow(Z, cmap="binary", aspect="auto")
    fig.colorbar(heatmap, ax=ax, shrink=0.5, aspect=5)

    ax.set_title(f"Norm of Z: {norm_Z:.2f}", pad=20)

    # Save the heatmap to a file
    id = str(uuid.uuid4())
    file_name = f"temp/{id}.png"
    plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Read the saved image
    plot_img = cv2.imread(file_name)
    os.remove(file_name)
    plot_img = cv2.resize(
        plot_img, (img_size, img_size)
    )  # Resize to match frame height
    return plot_img


def normalize_tensor(tensor):
    norm = torch.norm(tensor, p=2)  # Compute L2 norm
    if norm.item() != 0:  # Check if norm is not zero to avoid division by zero
        tensor.data /= norm
    return tensor


class SplitLASSO(BasePolicy):
    def __init__(
        self,
        feaNet: ConvNetwork,
        feature_weights: torch.Tensor,
        a_dim: int,
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
        super(SplitLASSO, self).__init__()

        ### constants
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
        self.feature_weights = nn.Parameter(feature_weights).to(
            dtype=self._dtype, device=self.device
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

    def get_features(self, obs, to_numpy: bool = False):
        obs = self.preprocess_obs(obs)
        with torch.no_grad():
            phi, _ = self.feaNet(obs["observation"], deterministic=True)
        if to_numpy:
            phi = phi.cpu().numpy()
        return phi

    def get_cumulative_features(self, obs, to_numpy: bool = False):
        """
        The naming intuition is that phi and psi are not really distinguishable
        """
        obs = self.preprocess_obs(obs)
        with torch.no_grad():
            phi, _ = self.feaNet(
                obs["observation"], obs["agent_pos"], deterministic=True
            )
            psi, _ = self.psiNet(phi)

        if to_numpy:
            psi = psi.cpu().numpy()
        return psi, {}

    def phi_Loss(self, states, actions, next_states, rewards):
        """
        Training target: phi_r (reward), phi_s (state)  -->  (Critic: feaNet)
        Method: reward mse (r - phi_r * w), state_pred mse (s' - D(phi_s, a))
        ---------------------------------------------------------------------------
        phi ~ [N, F/2]
        w ~ [1, F/2]
        """
        phi, conv_dict = self.feaNet(states, deterministic=False)

        phi_r, phi_s = self.split(phi)

        reward_pred = self.multiply_weights(phi_r, self.feature_weights)
        reward_loss = self._reward_loss_scaler * self.mse_loss(reward_pred, rewards)

        state_pred = self.decode(phi_s, actions, conv_dict)
        state_loss = self._state_loss_scaler * (
            1 / self.mse_loss(state_pred, next_states)
        )

        weight_norm = 0
        for param in self.feaNet.parameters():
            if param.requires_grad:  # Only include parameters that require gradients
                weight_norm += torch.norm(param, p=2)  # L

        weight_loss = self._weight_loss_scaler * weight_norm

        phi_norm = torch.norm(phi, p=1)
        lasso_loss = self._lasso_loss_scaler * phi_norm

        lasso_penalty = torch.relu(1e-3 - phi_norm)

        phi_loss = reward_loss + state_loss + weight_loss + lasso_loss + lasso_penalty

        # Plot predicted vs true rewards
        if self._forward_steps % 10 == 0:
            self.plot_rewards(reward_pred, rewards)

        return phi_loss, {
            "reward_loss": reward_loss,
            "state_loss": state_loss,
            "weight_loss": weight_loss,
            "lasso_loss": lasso_loss,
            "phi_norm": phi_norm,
        }

    def decode(self, phi, actions, conv_dict):
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

        reconstructed_state = self.feaNet.decode(phi, actions, conv_dict)
        return reconstructed_state

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
            "SF/lasso_loss": phi_loss_dict["lasso_loss"].item(),
            "SF/phi_norm": phi_loss_dict["phi_norm"].item(),
        }
        loss_dict.update(norm_dict)
        loss_dict.update(phi_grad_dict)

        t1 = time.time()
        self.eval()
        return loss_dict, t1 - t0

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
        feature_weights = self.feature_weights.clone().cpu()

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