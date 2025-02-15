import os
import cv2
import uuid
import math
import random
import torch
import json
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans
from utils.get_args import get_args
from torch.utils.tensorboard import SummaryWriter
from log.wandb_logger import WandbLogger


def override_args(env_name: str | None = None):
    args = get_args(verbose=False)
    if env_name is not None:
        args.env_name = env_name
    file_path = "assets/env_params.json"
    current_params = load_hyperparams(file_path=file_path, env_name=args.env_name)

    # use pre-defined params if no pram given as args
    for k, v in current_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


def load_hyperparams(file_path, env_name):
    """Load hyperparameters for a specific environment from a JSON file."""
    try:
        with open(file_path, "r") as f:
            hyperparams = json.load(f)
            return hyperparams.get(env_name, {})
    except FileNotFoundError:
        print(
            f"No file found at {file_path}. Returning default empty dictionary for {env_name}."
        )
        return {}


def separate_trajectories(features, terminals):
    terminal_indices = np.where(terminals == 1)[0]
    trajectories = []
    start_idx = 0
    for end_idx in terminal_indices:
        trajectories.append(
            features[start_idx : end_idx + 1]
        )  # Include the terminal state
        start_idx = end_idx + 1
    if start_idx < len(features):
        trajectories.append(features[start_idx:])
    return trajectories


def average_trajectories(trajectories):
    # Find the maximum trajectory length
    max_len = max(traj.shape[0] for traj in trajectories)
    feature_dim = trajectories[0].shape[1]

    # Initialize a padded 3D array
    padded_trajectories = np.zeros((len(trajectories), max_len, feature_dim))

    # Fill the padded_trajectories array and keep track of valid lengths
    for i, traj in enumerate(trajectories):
        padded_trajectories[i, : traj.shape[0], :] = traj

    # Compute the sum and valid counts for each time step
    traj_count = np.zeros((max_len, feature_dim))
    for i, traj in enumerate(trajectories):
        traj_count[
            : traj.shape[0], :
        ] += 1  # Increment count only where valid data exists

    # Calculate the average, taking into account the number of valid entries
    avg_trajectory = np.sum(padded_trajectories, axis=0) / traj_count

    return avg_trajectory


def seed_all(seed=0):
    # Set the seed for hash-based operations in Python
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Set the seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups

    # Ensure reproducibility of PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_model_summary(model, model_name="Model"):
    # Header with model name
    print("=" * 50)
    print(f"{model_name:<30} {'Param # (K)':>15}")
    print("=" * 50)

    total_params = 0
    total_trainable_params = 0
    total_non_trainable_params = 0

    # Iterate through model layers
    for name, param in model.named_parameters():
        num_params = np.prod(param.size())
        total_params += num_params

        if param.requires_grad:
            total_trainable_params += num_params
        else:
            total_non_trainable_params += num_params

        # # Layer name and number of parameters (in thousands)
        # print(f"{name:<30} {num_params / 1e3:>15,.2f} K")

    # Footer with totals
    # print("=" * 50)
    print(f"Total Parameters: {total_params / 1e3:,.2f} K")
    print(f"Trainable Parameters: {total_trainable_params / 1e3:,.2f} K")
    print(f"Non-trainable Parameters: {total_non_trainable_params / 1e3:,.2f} K")
    print("=" * 50)


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


def calculate_output_size(input_size, kernel_size, stride, padding):
    """Calculates the output size after a Conv2D or Pooling layer."""
    return math.floor((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)


def check_output_padding_needed(layers, input_size, verbose=False):
    """Checks if output padding is needed for ConvTranspose2D given a list of layers."""
    current_size = input_size
    layer_index = 0
    conv_transpose_needed = []

    for layer in layers:
        if layer["type"] == "conv" or layer["type"] == "pool":
            # Calculate the output size after this layer
            new_size = calculate_output_size(
                current_size, layer["kernel_size"], layer["stride"], layer["padding"]
            )

            # Check if ConvTranspose2D would need output padding to restore original size
            expected_input_size = (
                (new_size - 1) * layer["stride"]
                - 2 * layer["padding"]
                + layer["kernel_size"]
            )
            output_padding = current_size - expected_input_size

            # Store if output_padding is required (i.e., the size doesn't match perfectly)
            if layer["type"] == "conv":
                conv_transpose_needed.append(
                    {
                        "layer_index": layer_index,
                        "output_padding_needed": output_padding != 0,
                        "output_padding": max(0, output_padding),
                        "original_size": current_size,
                        "new_size": new_size,
                    }
                )

            # Update the current size
            current_size = new_size
        layer_index += 1

        if verbose:
            for result in conv_transpose_needed:
                print(f"Layer {result['layer_index']}:")
                print(f"  Original size: {result['original_size']}")
                print(f"  New size: {result['new_size']}")
                if result["output_padding_needed"]:
                    print(f"  Output padding needed: {result['output_padding']}")
                else:
                    print("  No output padding needed")

    return conv_transpose_needed


def calculate_flatten_size(input_shape, conv_layers):
    """
    Calculate the flattened size of the data after a series of convolutional and pooling layers.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        conv_layers (list of dicts): A list where each entry represents a layer in the network.
                                     Each entry is a dictionary with the following keys:
                                      - "type": Either "conv" or "pool"
                                      - "kernel_size": The size of the kernel (int or tuple)
                                      - "stride": The stride of the convolution or pooling (int or tuple)
                                      - "padding": The padding (int or tuple)
                                      - "filters": Number of filters for convolutional layers (ignored for pooling layers)

    Returns:
        final_size (int): Flattened size of the feature map after all layers.
        output_shape (tuple): Final shape (height, width, channels).
    """
    height, width, channels = input_shape

    for layer in conv_layers:
        kernel_size = layer["kernel_size"]
        stride = layer["stride"]
        padding = layer["padding"]

        # Handle kernel_size, stride, and padding being either tuples or ints
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        if layer["type"] == "conv":
            # Update height and width after convolution
            height = (
                math.floor((height - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
            )
            width = (
                math.floor((width - kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
            )

            channels = layer[
                "out_filters"
            ]  # Number of filters changes the depth (channels)

        elif layer["type"] == "pool":
            # Update height and width after pooling
            height = (
                math.floor((height - kernel_size[0] + 2 * padding[0]) / stride[0]) + 1
            )
            width = (
                math.floor((width - kernel_size[1] + 2 * padding[1]) / stride[1]) + 1
            )

    # The flattened size is height * width * channels
    final_size = height * width * channels
    output_shape = (height, width, channels)

    return final_size, output_shape


def save_dim_to_args(env, args):
    """
    Note: 1) One may drop the unneseccary actions to speed up the learning
            by constraining the action to any number (e.g., a ~ [0, 3])
          2) seeding would not matter

        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3

        # Drop an object
        drop = 4

        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6
    """
    args.s_dim = env.observation_space.shape  # (width, height, colors)
    args.flat_s_dim = int(np.prod(args.s_dim))
    if isinstance(env.action_space, gym.spaces.Discrete):
        args.a_dim = int(env.action_space.n)
    elif isinstance(env.action_space, gym.spaces.Box):
        args.a_dim = int(env.action_space.shape[0])
    args.grid_size = int(args.s_dim[0])

    print(f"Problem dimension (|S|/|A|): {args.s_dim}/{args.a_dim}")
    env.close()


def setup_logger(args, unique_id, exp_time, seed):
    """
    setup logger both using WandB and Tensorboard
    Return: WandB logger, Tensorboard logger
    """
    # Get the current date and time
    now = datetime.now()
    args.running_seed = seed

    if args.group is None:
        args.group = "-".join((exp_time, unique_id))

    if args.name is None:
        args.name = "-".join(
            (args.algo_name, args.env_name, unique_id, "seed:" + str(seed))
        )

    if args.project is None:
        args.project = args.env_name

    args.logdir = os.path.join(args.logdir, args.group)

    default_cfg = vars(args)
    logger = WandbLogger(
        config=default_cfg,
        project=args.project,
        group=args.group,
        name=args.name,
        log_dir=args.logdir,
        log_txt=True,
        fps=args.render_fps,
    )
    logger.save_config(default_cfg, verbose=args.verbose)

    tensorboard_path = os.path.join(logger.log_dir, "tensorboard")
    os.mkdir(tensorboard_path)
    writer = SummaryWriter(log_dir=tensorboard_path)

    return logger, writer


def estimate_advantages(
    rewards, terminals, values, gamma=0.99, gae=0.95, device=torch.device("cpu")
):
    rewards, terminals, values = (
        rewards.to(torch.device("cpu")),
        terminals.to(torch.device("cpu")),
        values.to(torch.device("cpu")),
    )
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * (1 - terminals[i]) - values[i]
        advantages[i] = deltas[i] + gamma * gae * prev_advantage * (1 - terminals[i])

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    # advantages = (advantages - advantages.mean()) / advantages.std()
    advantages, returns = advantages.to(device), returns.to(device)
    return advantages, returns


def estimate_psi(phi, terminals, gamma, device=torch.device("cpu"), mode="numpy"):
    """
    Estimates psi using either PyTorch or NumPy.

    Args:
        phi: Torch tensor or NumPy array of features.
        terminals: Torch tensor or NumPy array indicating terminal states.
        gamma: Discount factor.
        device: Torch device (only used in torch mode).
        mode: "torch" for PyTorch version, "numpy" for NumPy version.

    Returns:
        psi: Computed psi values as a Torch tensor or NumPy array.
    """
    if mode == "torch":
        # Ensure tensors are on the CPU for computation
        phi, terminals = phi.to("cpu"), terminals.to("cpu")

        # Initialize psi
        psi = torch.zeros_like(phi)

        prev_psi = 0
        for i in reversed(range(phi.size(0))):
            psi[i] = phi[i] + gamma * prev_psi * (1 - terminals[i])
            prev_psi = psi[i]

        # Return psi to the requested device
        return psi.to(device)

    elif mode == "numpy":
        # Ensure inputs are NumPy arrays
        if isinstance(phi, torch.Tensor):
            phi = phi.cpu().numpy()
        if isinstance(terminals, torch.Tensor):
            terminals = terminals.cpu().numpy()

        # Initialize psi
        psi = np.zeros_like(phi)

        prev_psi = 0
        for i in reversed(range(phi.shape[0])):
            psi[i] = phi[i] + gamma * prev_psi * (1 - terminals[i])
            prev_psi = psi[i]

        return psi

    else:
        raise ValueError("Invalid mode. Choose 'torch' or 'numpy'.")
