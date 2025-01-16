import os
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

from utils.wrappers import GridWrapper


def get_grid_tensor(env, grid_type):
    """
    Can be extended to the multigrid by removing multiple agents
    FourRoom and LavaRoom tailored
    """
    obs, _ = env.reset(seed=grid_type)
    grid_tensor = obs["observation"]
    env.close()

    if isinstance(env, GridWrapper):
        agent_loc = np.where(grid_tensor[:, :, 0] == 10)
        grid_tensor[agent_loc[0], agent_loc[1], 0] = 1

        x_coords, y_coords = np.where(
            (grid_tensor[:, :, 0] != 2)
            & (grid_tensor[:, :, 0] != 8)
            & (grid_tensor[:, :, 0] != 9)
        )  # find idx where not wall

    else:
        raise NotImplementedError(f"Not implemented for {env}")
    return grid_tensor, (x_coords, y_coords)


def generate_possible_tensors(env, path, args, tile_size, grid_type):
    # Check if the given render mode is not human
    if env.render_mode != "rgb_array":
        raise ValueError(f"render mode should be rgb_array. Current: {env.render_mode}")

    # get the raw grid tensor without agent in the image
    grid_tensor, original_tensor = get_grid_tensor(env, grid_type)
    # get coordinates where the agent can visit
    x_coords, y_coords = np.where(
        grid_tensor[:, :, 0] != 2 and grid_tensor[:, :, 0] != 8
    )  # find idx where not wall

    # there are four possible directions
    # this should be changed in another environmental domains

    # create a path for saving each directional map separately
    path = os.path.join(path, "allStates")
    os.mkdir(path)

    total_iterations = len(x_coords)  # Total number of iterations

    # Progress bar with trange
    with trange(total_iterations, desc="Generating all possible state images") as pbar:
        for x, y in zip(x_coords, y_coords):
            # Render the new state
            temp_img = grid_tensor.copy()
            temp_img[x, y, 0] = 10
            img = temp_img.copy()  # * 20
            img = np.repeat(np.repeat(img, tile_size, axis=0), tile_size, axis=1)
            img = np.squeeze(img)  # (n, n) 2d image for grey scale saving

            del temp_img

            # Save the image
            plt.imsave(
                os.path.join(path, f"{str(x)}_{str(y)}.png"),
                img,
                cmap="gray",
            )
            plt.close()
            pbar.update(1)  # Update the progress bar by 1 for each iteration

    args.path_allStates = path
    return original_tensor, (x_coords, y_coords)
