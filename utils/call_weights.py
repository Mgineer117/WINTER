import numpy as np
from math import ceil, floor
import torch
import torch.nn as nn
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from models.policy import SF_LASSO
from sklearn.cluster import KMeans

from utils.buffer import TrajectoryBuffer
from utils.sampler import OnlineSampler
from utils.utils import estimate_psi


def call_options(
    algo_name: str,
    sf_dim: int,
    snac_split_ratio: float,
    temporal_balance_ratio: float,
    num_options: int,
    sf_network: SF_LASSO,
    sampler: OnlineSampler,
    buffer: TrajectoryBuffer,
    DIF_batch_size: int,
    grid_type: int,
    gamma: float,
    method: str = "top",
    device: torch.device = torch.device("cpu"),
):
    # required params
    if algo_name == "SNAC":
        num_r_features = floor(sf_dim * snac_split_ratio)
        num_s_features = sf_dim - num_r_features
        # collecting num // 2 for each of reward and state features
        num_options = num_options // 2
    else:
        num_r_features = 0
        num_s_features = sf_dim

    # Warm buffer
    buffer.max_batch_size = DIF_batch_size
    buffer = warm_buffer(sf_network, sampler, buffer, grid_type)
    samples = buffer.sample_all()

    # collect samples
    states = torch.from_numpy(samples["states"]).to(device)
    terminals = samples["terminals"]

    # get features (phi and psi)
    with torch.no_grad():
        features = sf_network.get_features(states, deterministic=True, to_numpy=True)
    psi = estimate_psi(features, terminals=terminals, gamma=gamma)

    # prepare for subtask discovery
    if num_r_features > 0:
        phi_R, phi_S = psi[:, :num_r_features], psi[:, num_r_features:]
        psi_R, psi_S = psi[:, :num_r_features], psi[:, num_r_features:]

        _, S_R, V_R = np.linalg.svd(psi_R)
        _, S_S, V_S = np.linalg.svd(psi_S)
        subtask_vectors = {"rewards": V_R, "states": V_S}
    else:
        phi_S = features
        _, S_S, V_S = np.linalg.svd(psi)
        subtask_vectors = {"rewards": None, "states": V_S}

    for key, subtask_vector in subtask_vectors.items():
        if key == "rewards":
            if subtask_vector is not None:
                V = subtask_vector[:num_options]
            else:
                V = None
        else:
            if method == "top":
                V = subtask_vector[:num_options]
            elif method == "cvs":
                # Use K-Means++ to cluster V (rows)
                kmeans = KMeans(
                    n_clusters=num_options, init="k-means++", random_state=42
                )
                kmeans.fit(subtask_vector)
                centroids = kmeans.cluster_centers_

                V = centroids[:num_options]
            elif method == "crs":
                # Use K-Means++ to cluster V.T (columns/features)
                kmeans = KMeans(
                    n_clusters=num_options, init="k-means++", random_state=42
                )
                kmeans.fit(subtask_vector @ phi_S.T)
                cluster_labels = kmeans.labels_

                crs_V = np.empty((num_options, subtask_vector.shape[-1]))
                for i in range(num_options):
                    crs_V[i] = np.mean(subtask_vector[cluster_labels == i], axis=0)

                V = crs_V

            elif method == "trs":
                # Collect n% of top n and remainder CRS clustered
                n = ceil(temporal_balance_ratio * num_options)
                top_n_V = subtask_vector[:n]
                remainder_V = subtask_vector[n:]
                pseudo_rewards = remainder_V @ phi_S.T

                kmeans = KMeans(
                    n_clusters=num_options - n, init="k-means++", random_state=42
                )
                kmeans.fit(pseudo_rewards)
                cluster_labels = kmeans.labels_

                crs_V = np.empty((num_options - n, remainder_V.shape[-1]))
                for i in range(num_options - n):
                    crs_V[i] = np.mean(remainder_V[cluster_labels == i], axis=0)

                V = np.concatenate((top_n_V, crs_V), axis=0)
            else:
                raise ValueError(f"method {method} not recognized")

        # Include both the original and negative counterparts
        if V is not None:
            if key == "rewards":
                reward_options = np.concatenate((V, -V), axis=0)
            else:
                state_options = np.concatenate((V, -V), axis=0)
        else:
            if key == "rewards":
                reward_options = None
            else:
                # shouldn't be None
                state_options = None

    return reward_options, state_options


def get_reward_maps(
    env: gym.Env, sf_network: nn.Module, V: list, feature_dim: int, grid_type: int
):

    raw_grid, x_coords, y_coords = get_grid_and_coords(env, grid_type)
    random_indices = random.sample(range(len(x_coords)), 2)
    # fix enemy assignment
    # raw_grid[x_coords[random_indices[0]], y_coords[random_indices[0]], 1:] = 2
    # raw_grid[x_coords[random_indices[1]], y_coords[random_indices[1]], 1:] = 2

    raw_grid[3, 3, 1:] = 2
    raw_grid[8, 8, 1:] = 2

    # find idx where not wall and red agent
    pos = np.where(
        (raw_grid[:, :, 0] != 0)
        & (raw_grid[:, :, 1] != 2)
        & (raw_grid[:, :, 1] != 3)
        & (raw_grid[:, :, 1] != 4)
    )

    img = plotRewardMap(
        sf_network=sf_network,
        raw_grid=raw_grid,
        V=V,
        feature_dim=feature_dim,
        coords=pos,
    )
    return img


def get_grid_and_coords(env, grid_type):
    obs, _ = env.reset(seed=grid_type)
    raw_grid = obs["observation"]
    env.close()

    agent_pos = np.where(raw_grid[:, :, 1] == 1)
    enemy_pos = np.where(raw_grid[:, :, 1] == 2)

    raw_grid[agent_pos[0], agent_pos[1], 1] = 0
    raw_grid[agent_pos[0], agent_pos[1], 2] = 0

    raw_grid[enemy_pos[0], enemy_pos[1], 1] = 0
    raw_grid[enemy_pos[0], enemy_pos[1], 2] = 0

    x_coords, y_coords = np.where(
        (raw_grid[:, :, 0] != 0) & (raw_grid[:, :, 1] != 3) & (raw_grid[:, :, 1] != 4)
    )  # find idx where not wall

    return raw_grid, x_coords, y_coords


def plotRewardMap(
    sf_network: nn.Module,
    raw_grid: np.ndarray,
    V: list,
    feature_dim: int,
    coords: tuple,
):
    # convert V to tensor
    for i, vector in enumerate(V):
        if vector is not None:
            V[i] = torch.from_numpy(vector).to(torch.float32)

    ### Load parameters
    x_grid_dim, y_grid_dim, _ = raw_grid.shape
    num_reward_options = V[0].shape[0] if V[0] is not None else 0
    num_state_options = V[1].shape[0] if V[1] is not None else 0
    agent_dirs = [0, 1, 2, 3]
    num_vec = num_reward_options + num_state_options
    num_r_features = sf_network.num_r_features

    x_coords, y_coords = coords
    device_of_model = next(sf_network.parameters()).device

    # create a placeholder
    features = torch.zeros(x_grid_dim, y_grid_dim, feature_dim)
    deltaPhi = torch.zeros(len(agent_dirs), x_grid_dim, y_grid_dim, feature_dim)

    # will avg across agent_dirs
    rewards = torch.zeros(num_vec, x_grid_dim, y_grid_dim)

    for x, y in zip(x_coords, y_coords):
        # # Load the image as a NumPy array
        img = raw_grid.copy()
        img[x, y, 1] = 1
        img[x, y, 2] = 2

        with torch.no_grad():
            img = torch.from_numpy(img).to(torch.float32).to(device_of_model)
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            phi = sf_network.get_features(img, deterministic=True)
        features[x, y, :] = phi

    # ### COMPUTE DELTA-PHI
    # coordinates = np.stack((coords[0], coords[1]), axis=-1)
    # for agent_dir in agent_dirs:
    #     """
    #     agent_dir 0: left
    #     agent_dir 1: up
    #     agent_dir 2: right
    #     agent_dir 3: down
    #     """
    #     for x, y in zip(coords[0], coords[1]):
    #         if agent_dir == 0:
    #             temp_x, temp_y = x, y - 1
    #             if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
    #                 deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
    #             else:
    #                 deltaPhi[agent_dir, x, y, :] += features[x, y, :]
    #         elif agent_dir == 1:
    #             temp_x, temp_y = x - 1, y
    #             if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
    #                 deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
    #             else:
    #                 deltaPhi[agent_dir, x, y, :] += features[x, y, :]
    #         elif agent_dir == 2:
    #             temp_x, temp_y = x, y + 1
    #             if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
    #                 deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
    #             else:
    #                 deltaPhi[agent_dir, x, y, :] += features[x, y, :]
    #         elif agent_dir == 3:
    #             temp_x, temp_y = x + 1, y
    #             if any((coordinates == (temp_x, temp_y)).all(axis=-1)):
    #                 deltaPhi[agent_dir, x, y, :] += features[temp_x, temp_y, :]
    #             else:
    #                 deltaPhi[agent_dir, x, y, :] += features[x, y, :]

    # # sum all connected next_phi - current phi
    # deltaPhi = torch.mean(deltaPhi, axis=0)  # [x, y, f]
    # deltaPhi -= features
    deltaPhi = features

    r_deltaPhi, s_deltaPhi = (
        deltaPhi[:, :, :num_r_features],
        deltaPhi[:, :, num_r_features:],
    )

    for vec_idx in range(num_vec):
        is_reward_option = True if vec_idx < num_reward_options else False
        if is_reward_option:
            idx = vec_idx
            reward = torch.sum(
                torch.mul(r_deltaPhi[:, :, :], V[0][idx, :]),
                axis=-1,
            )
        else:
            idx = vec_idx - num_reward_options
            reward = torch.sum(
                torch.mul(s_deltaPhi[:, :, :], V[1][idx, :]),
                axis=-1,
            )

        for x, y in zip(x_coords, y_coords):
            rewards[vec_idx, x, y] += reward[x, y]

    ### Normalization
    for k in range(num_vec):
        # Identify positive and negative rewards
        pos_rewards = rewards[k, :, :] > 0
        neg_rewards = rewards[k, :, :] < 0

        # Normalize positive rewards to the range [0, 1]
        if pos_rewards.any():  # Check if there are any positive rewards
            r_pos_max = rewards[k, pos_rewards].max()
            r_pos_min = rewards[k, pos_rewards].min()
            rewards[k, pos_rewards] = (rewards[k, pos_rewards] - r_pos_min) / (
                r_pos_max - r_pos_min + 1e-10
            )
        # Normalize negative rewards to the range [-1, 0]
        if neg_rewards.any():  # Check if there are any negative rewards
            r_neg_max = rewards[k, neg_rewards].max()  # Closest to 0 (least negative)
            r_neg_min = rewards[k, neg_rewards].min()  # Most negative
            rewards[k, neg_rewards] = (rewards[k, neg_rewards] - r_neg_max) / (
                r_neg_max - r_neg_min + 1e-10
            )

    # Smoothing the tensor using a uniform filter
    rewards = rewards.numpy()
    # for k in range(rewards.shape[0]):
    #     rewards[k, :, :] = uniform_filter(rewards[k, :, :], size=3)

    # Define a custom colormap with black at the center
    colors = [
        (0.2, 0.2, 1),
        (0.2667, 0.0039, 0.3294),
        (1, 0.2, 0.2),
    ]  # Blue -> Black -> Red
    cmap = mcolors.LinearSegmentedColormap.from_list("pale_blue_dark_pale_red", colors)

    images = []
    walls = np.where(
        (raw_grid[:, :, 0] == 0)
        | (raw_grid[:, :, 1] == 2)
        | (raw_grid[:, :, 1] == 3)
        | (raw_grid[:, :, 1] == 4)
    )
    for i in range(num_vec):
        grid = np.zeros((x_grid_dim, y_grid_dim))
        grid += rewards[i, :, :]

        if np.sum(np.int8(grid > 0)) == 0:
            for x, y in zip(coords[0], coords[1]):
                grid[x, y] += 1.0
        grid[walls] = 0.0

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 8))
        ax0.imshow(raw_grid * 50)
        ax1.imshow(grid, cmap=cmap, interpolation="nearest", vmin=-1, vmax=1)
        plt.tight_layout()

        # Render the figure to a canvas
        canvas = FigureCanvas(fig)
        canvas.draw()

        # Convert canvas to a NumPy array
        reward_img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        reward_img = reward_img.reshape(
            canvas.get_width_height()[::-1] + (3,)
        )  # Shape: (height, width, 3)
        plt.close()
        images.append(reward_img)
        i += 1
    return images


def warm_buffer(
    sf_network: nn.Module,
    sampler: OnlineSampler,
    buffer: TrajectoryBuffer,
    grid_type: int,
):
    # make sure there is nothing there
    buffer.wipe()

    # collect enough batch
    count = 0
    total_sample_time = 0
    sample_time = 0
    while buffer.num_samples < buffer.max_batch_size:
        batch, sampleT = sampler.collect_samples(sf_network, grid_type=grid_type)
        buffer.push(batch)
        sample_time += sampleT
        total_sample_time += sampleT
        if count % 25 == 0:
            print(
                f"\nWarming buffer {buffer.num_samples}/{buffer.max_batch_size} | sample_time = {sample_time:.2f}s",
                end="",
            )
            sample_time = 0
        count += 1
    print(
        f"\nWarming Complete! {buffer.num_samples}/{buffer.max_batch_size} | total sample_time = {total_sample_time:.2f}s",
        end="",
    )
    print()

    return buffer


# reward_feature_weights, reward_options, state_options = call_feature_weights(3, 5)
