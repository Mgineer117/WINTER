import numpy as np


class ObservationNormalizer:
    def __init__(
        self,
        state_dim,
        mode="ema",
        alpha=0.95,
        window_size=500,
        weights=None,
        max_updates=1000,
    ):
        """
        mode: Type of moving average ('ema', 'cma', or 'wma')
        alpha: Decay factor for EMA
        window_size: Size of the window for WMA
        weights: Custom weights for WMA
        max_updates: Maximum number of updates allowed for normalization
        state_dim: The state dimension (shape of the observation, e.g., (batch_size, height, width, channels))
        """
        self.mode = mode
        self.alpha = alpha
        self.window_size = window_size
        self.max_updates = max_updates  # Maximum number of updates allowed
        self.update_count = 0  # Track the number of updates

        # Initialize the statistics
        self.mean = np.zeros(
            state_dim[1:]
        )  # Initialize mean with the state dimension (excluding batch size)
        self.var = np.ones(state_dim[1:])  # Initialize variance with ones
        self.count = 0

        # Initialize based on the mode
        if mode == "ema":
            self.normalizer = EMA(alpha=self.alpha, state_dim=state_dim)
        elif mode == "cma":
            self.normalizer = CMA(state_dim=state_dim)
        elif mode == "wma":
            self.normalizer = WMA(
                window_size=self.window_size, weights=weights, state_dim=state_dim
            )
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented.")

    def normalize(self, observation):
        """
        Normalize the observation using the selected moving average mode.
        If the observation is a batch, apply normalization per observation.
        """
        # Stop updating once the max updates are reached
        if self.update_count >= self.max_updates:
            return (observation - self.mean) / (np.sqrt(self.var) + 1e-8)

        # Update the statistics based on the selected mode
        if self.mode == "ema":
            self.normalizer.update(observation)
            self.mean = self.normalizer.mean
            self.var = self.normalizer.variance
        elif self.mode == "cma":
            self.normalizer.update(observation)
            self.mean = self.normalizer.mean
            self.var = np.zeros_like(self.mean)  # CMA doesn't maintain variance
        elif self.mode == "wma":
            self.normalizer.update(observation)
            self.mean = self.normalizer.mean
            self.var = np.zeros_like(self.mean)  # WMA doesn't maintain variance for now

        self.update_count += 1

        # Return the normalized observation after the update
        return (observation - self.mean) / (np.sqrt(self.var) + 1e-8)


class EMA:
    def __init__(self, alpha=0.99, state_dim=None):
        """
        alpha: Decay factor for the exponential moving average.
        state_dim: The state dimension, e.g., (batch_size, height, width, channels)
        """
        self.alpha = alpha
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim[1:])  # Initialize mean based on state dimension
        self.variance = np.ones(state_dim[1:])  # Initialize variance with ones

    def update(self, observation):
        """
        Update the moving averages of mean and variance using the EMA formula.
        """
        if self.mean is None:
            self.mean = observation
            self.variance = np.zeros_like(observation)
        else:
            # Update mean and variance using EMA formula
            self.mean = self.alpha * self.mean + (1 - self.alpha) * observation
            self.variance = (
                self.alpha * self.variance
                + (1 - self.alpha) * (observation - self.mean) ** 2
            )

    def normalize(self, observation):
        """
        Normalize the observation using the current moving averages of mean and variance.
        """
        self.update(observation)
        return (observation - self.mean) / (np.sqrt(self.variance) + 1e-8)


class CMA:
    def __init__(self, state_dim=None):
        """
        Cumulative moving average (CMA) for mean calculation.
        state_dim: The state dimension (shape of the observation, e.g., (batch_size, height, width, channels))
        """
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim[1:])  # Initialize mean based on state dimension
        self.count = 0

    def update(self, observation):
        """
        Update the cumulative moving average of the observations.
        """
        self.count += 1
        if self.mean is None:
            self.mean = observation
        else:
            # Update cumulative mean using CMA formula
            self.mean = (self.mean * (self.count - 1) + observation) / self.count

    def normalize(self, observation):
        """
        Normalize the observation using the cumulative mean.
        """
        self.update(observation)
        return observation - self.mean


class WMA:
    def __init__(self, window_size=100, weights=None, state_dim=None):
        """
        window_size: Size of the window for weighted moving average
        weights: Custom weights for WMA.
        state_dim: The state dimension (shape of the observation, e.g., (batch_size, height, width, channels))
        """
        self.window_size = window_size
        self.weights = weights if weights is not None else np.ones(window_size)
        self.state_dim = state_dim
        self.window = []
        self.count = 0
        self.sum = np.zeros(state_dim[1:])  # Initialize sum to state dimension

    def update(self, observation):
        """
        Update the weighted moving average with the current observation.
        """
        self.window.append(observation)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        # Update weighted sum
        weighted_sum = np.dot(self.weights[: len(self.window)], self.window)
        self.sum = weighted_sum
        self.count = len(self.window)
        self.mean = self.sum / self.count  # Weighted mean for normalization

    def normalize(self, observation):
        """
        Normalize the observation using the weighted moving average mean.
        """
        self.update(observation)
        return observation - self.mean


# Example usage outside of class to track mean and variance difference with 4D tensor

# Initialize the normalizer with state dimension (batch_size, height, width, channels)
state_dim = (
    3,
    5,
    2,
)  # Example state dimension (batch_size=5, height=3, width=3, channels=2)
normalizer = ObservationNormalizer(
    mode="ema", alpha=0.9, max_updates=10, state_dim=state_dim
)

# Simulate a sequence of 4D observations (e.g., 5 samples of 3x3 images with 2 channels)
observations = np.random.uniform(
    0, 10, size=(10, 3, 5, 2)
)  # Shape: (batch_size, height, width, channels)

prev_mean = None
prev_var = None

for obs in observations:
    normalized_obs = normalizer.normalize(obs)

    # Print the difference in mean and variance after each update
    if prev_mean is not None:
        print(normalizer.mean)
        mean_diff = np.round(np.abs(normalizer.mean - prev_mean), 2)
        var_diff = np.round(np.abs(normalizer.var - prev_var), 2)
        print(
            f"Update {normalizer.update_count}: Mean diff = {mean_diff}, Var diff = {var_diff}"
        )

    prev_mean = normalizer.mean
    prev_var = normalizer.var

normalized_obs = normalizer.normalize(observations)
print(normalized_obs)
