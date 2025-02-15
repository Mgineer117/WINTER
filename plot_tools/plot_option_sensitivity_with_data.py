import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Define n (even number)
data = [
    0.20010867681177563,
    0.06625276160606511,
    0.1419886745653092,
    0.22849311325949745,
    0.16791195333235304,
    0.2882266065093388,
    0.1807019784188955,
    0.13629395952350565,
    0.1834614229567018,
    0.04354979391959854,
    0.10053236677386536,
    0.05225412096387955,
    0.15750608768423344,
    0.04288884294694657,
    0.060555457433088655,
    0.05297013719875651,
]
n = len(data)
half_n = n // 2

# Generate synthetic data
reward_options = data[:half_n]  # Mean differences for reward options
state_options = data[half_n:]  # Mean differences for state options

reward_options_mean = np.mean(reward_options)
state_options_mean = np.mean(state_options)

# Create index labels
indices_reward = np.arange(half_n)
indices_state = np.arange(half_n, n)

# Plotting
plt.figure(figsize=(5, 5))

# Scatter plot for reward options
plt.scatter(
    indices_reward,
    reward_options,
    color="blue",
    label="Reward Options",
    marker="o",
    s=100,
)

# Scatter plot for state options
plt.scatter(
    indices_state,
    state_options,
    color="red",
    label="State Options",
    marker="s",
    s=100,
)

# Add horizontal lines for means
plt.axhline(
    y=reward_options_mean,
    xmin=0,
    xmax=(half_n) / n,  # Normalize to [0,1] range for first half
    color="k",
    linestyle="-.",
    label="R-Mean",
)

plt.axhline(
    y=state_options_mean,
    xmin=(half_n) / n,  # Normalize for second half
    xmax=1,
    color="k",
    linestyle="--",
    label="S-Mean",
)

# Add x-ticks at each data point
plt.xticks(np.arange(n))

# Formatting
plt.xlabel("Index", fontsize=20)
plt.ylabel("Mean Reward Difference", fontsize=20)
plt.axvline(
    x=half_n - 0.5, color="k", linestyle="-", label="Split Line"
)  # Separate two groups

plt.legend(fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("option_sensitivity.png")
plt.show()
