import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from models.policy import LASSO
from sklearn.cluster import KMeans

from utils.buffer import TrajectoryBuffer
from utils.sampler import OnlineSampler
from utils.utils import estimate_psi


def create_matrix(n):
    # Initialize an empty list to hold the rows
    matrix = []

    # Dynamically generate the vectors
    for i in range(n):
        vector_positive = [0.0] * n
        vector_negative = [0.0] * n

        # Set the current index to +1.0 and -1.0 for positive and negative vectors
        vector_positive[i] = 1.0
        vector_negative[i] = -1.0

        # Add the vectors to the matrix
        matrix.append(vector_positive)
        matrix.append(vector_negative)

    # Convert the matrix to a NumPy array
    return np.array(matrix)


def call_feature_weights(sf_r_dim: int):
    """Create a feature weights for sf_r_dim and sf_s_dim

    Args:
        sf_r_dim (_type_): _description_
        sf_s_dim (_type_): _description_
    """

    if sf_r_dim == 3:
        reward_feature_weights = np.array([1.0, 0.0, -1.0])
    elif sf_r_dim == 5:
        reward_feature_weights = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    elif sf_r_dim == 7:
        reward_feature_weights = np.array([1.0, 0.66, 0.33, 0.0, -0.33, -0.66, -1.0])
    else:
        raise ValueError(f"sf_r_dim (given: {sf_r_dim}) is not one of [3, 5, 7] ")

    return reward_feature_weights  # , reward_options, state_options


def call_options(
    sf_r_dim: int,
    r_option_num: int,
    sf_s_dim: int,
    s_option_num: int,
    sf_network: LASSO,
    sampler: OnlineSampler,
    buffer: TrajectoryBuffer,
    grid_type: int,
    gamma: float,
    method: str = "top",
    device: torch.device = torch.device("cpu"),
):
    ### create a option vector
    reward_options = create_matrix(sf_r_dim)[: 2 * r_option_num]

    if sf_s_dim != 0:
        # Warm buffer
        buffer = warm_buffer(sf_network, sampler, buffer, grid_type)
        samples = buffer.sample_all()

        states = torch.from_numpy(samples["states"]).to(device)
        terminals = samples["terminals"]

        with torch.no_grad():
            features = sf_network.get_features(
                states, deterministic=True, to_numpy=True
            )
        _, features = sf_network.split(features, sf_network.sf_r_dim)
        psi = estimate_psi(features, terminals=terminals, gamma=gamma)

        _, S, V = np.linalg.svd(psi)

        if method == "top":
            V = V[:s_option_num]
        elif method == "cvs":
            # Use K-Means++ to cluster V (rows)
            kmeans = KMeans(n_clusters=s_option_num, init="k-means++", random_state=42)
            kmeans.fit(V)
            centroids = kmeans.cluster_centers_

            V = centroids[:s_option_num]
        elif method == "crs":
            # Use K-Means++ to cluster V.T (columns/features)
            kmeans = KMeans(n_clusters=s_option_num, init="k-means++", random_state=42)
            kmeans.fit(V @ features.T)
            cluster_labels = kmeans.labels_

            crs_V = np.empty((s_option_num, V.shape[-1]))
            for i in range(s_option_num):
                crs_V[i] = np.mean(V[cluster_labels == i], axis=0)

            V = crs_V

        elif method == "trs":
            n = int(0.25 * s_option_num)  # Calculate 25% of s_option_num
            top_n_V = V[:n]
            remainder_V = V[n:]

            kmeans = KMeans(
                n_clusters=s_option_num - n, init="k-means++", random_state=42
            )
            kmeans.fit(remainder_V @ features.T)
            cluster_labels = kmeans.labels_

            crs_V = np.empty((s_option_num - n, remainder_V.shape[-1]))
            for i in range(s_option_num - n):
                crs_V[i] = np.mean(remainder_V[cluster_labels == i], axis=0)

            V = np.concatenate((top_n_V, crs_V), axis=0)

        # Include both the original and negative counterparts
        state_options = np.concatenate((V, -V), axis=0)
    else:
        state_options = None

    return reward_options, state_options


def get_reward_maps(
    env: gym.Env, sf_network: nn.Module, V: list, feature_dim: int, grid_type: int
):

    grid_tensor, x_coords, y_coords = get_grid_and_coords(env, grid_type)
    random_indices = random.sample(range(len(x_coords)), 2)
    grid = grid_tensor.copy()
    # fix enemy assignment
    grid[x_coords[random_indices[0]], y_coords[random_indices[0]], 1:] = 2
    grid[x_coords[random_indices[1]], y_coords[random_indices[1]], 1:] = 2

    # find idx where not wall and red agent
    pos = np.where(
        (grid[:, :, 0] != 0)
        & (grid[:, :, 1] != 2)
        & (grid[:, :, 1] != 3)
        & (grid[:, :, 1] != 4)
    )

    img = plotRewardMap(
        sf_network=sf_network,
        grid_tensor=grid_tensor,
        V=V,
        feature_dim=feature_dim,
        coords=pos,
    )
    return img


def get_grid_and_coords(env, grid_type):
    obs, _ = env.reset(seed=grid_type)
    grid_tensor = obs["observation"]
    env.close()

    agent_pos = np.where(grid_tensor[:, :, 1] == 1)
    enemy_pos = np.where(grid_tensor[:, :, 1] == 2)

    grid_tensor[agent_pos[0], agent_pos[1], 1] = 0
    grid_tensor[agent_pos[0], agent_pos[1], 2] = 0

    grid_tensor[enemy_pos[0], enemy_pos[1], 1] = 0
    grid_tensor[enemy_pos[0], enemy_pos[1], 2] = 0

    x_coords, y_coords = np.where(
        (grid_tensor[:, :, 0] != 0)
        & (grid_tensor[:, :, 1] != 3)
        & (grid_tensor[:, :, 1] != 4)
    )  # find idx where not wall

    return grid_tensor, x_coords, y_coords


def plotRewardMap(
    sf_network: nn.Module,
    grid_tensor: np.ndarray,
    V: list,
    feature_dim: int,
    coords: tuple,
):
    # convert V to tensor
    for i, vector in enumerate(V):
        if vector is not None:
            V[i] = torch.from_numpy(vector).to(torch.float32)

    ### Load parameters
    x_grid_dim, y_grid_dim, _ = grid_tensor.shape
    num_reward_options = V[0].shape[0] if V[0] is not None else 0
    num_state_options = V[1].shape[0] if V[1] is not None else 0
    agent_dirs = [0, 1, 2, 3]
    num_vec = num_reward_options + num_state_options
    sf_r_dim = sf_network.sf_r_dim

    x_coords, y_coords = coords
    device_of_model = next(sf_network.parameters()).device

    # create a placeholder
    features = torch.zeros(x_grid_dim, y_grid_dim, feature_dim)
    deltaPhi = torch.zeros(len(agent_dirs), x_grid_dim, y_grid_dim, feature_dim)

    # will avg across agent_dirs
    rewards = torch.zeros(num_vec, x_grid_dim, y_grid_dim)

    for x, y in zip(x_coords, y_coords):
        # # Load the image as a NumPy array
        img = grid_tensor.copy()
        img[x, y, :] = 10  # 10 is an agent

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

    r_deltaPhi, s_deltaPhi = deltaPhi[:, :, :sf_r_dim], deltaPhi[:, :, sf_r_dim:]

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
        (grid_tensor[:, :, 0] == 0)
        | (grid_tensor[:, :, 1] == 2)
        | (grid_tensor[:, :, 1] == 3)
        | (grid_tensor[:, :, 1] == 4)
    )
    for i in range(num_vec):
        grid = np.zeros((x_grid_dim, y_grid_dim))
        grid += rewards[i, :, :]

        if np.sum(np.int8(grid > 0)) == 0:
            for x, y in zip(coords[0], coords[1]):
                grid[x, y] += 1.0
        grid[walls] = 0.0

        fig, ax = plt.subplots()
        ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=-1, vmax=1)
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
        batch, sampleT = sampler.collect_samples(
            sf_network, grid_type=grid_type, random_init_pos=True
        )
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
