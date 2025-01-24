import numpy as np


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


def call_feature_weights(sf_r_dim, sf_s_dim):
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

    ### create a option vector
    reward_options = create_matrix(sf_r_dim)
    state_options = create_matrix(sf_s_dim)

    return reward_feature_weights, reward_options, state_options


# reward_feature_weights, reward_options, state_options = call_feature_weights(3, 5)
