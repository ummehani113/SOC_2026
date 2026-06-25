# week3_compare.py

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Moving average function
# ----------------------------
def moving_average(data, window=10):
    """
    Computes the moving average of a list/array.
    This smooths the learning curve and makes trends easier to see.
    """
    data = np.array(data)

    if len(data) < window:
        return data

    return np.convolve(data, np.ones(window) / window, mode="valid")


# ----------------------------
# Load saved rewards
# ----------------------------
reinforce_returns = np.load("reinforce_returns.npy")
ppo_returns = np.load("ppo_returns.npy")


# ----------------------------
# Smooth the curves
# ----------------------------
reinforce_ma = moving_average(reinforce_returns, window=10)
ppo_ma = moving_average(ppo_returns, window=10)


# ----------------------------
# Plot comparison
# ----------------------------
plt.figure(figsize=(10, 6))

plt.plot(reinforce_ma, label="REINFORCE")
plt.plot(ppo_ma, label="PPO")

plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("REINFORCE vs PPO on CartPole-v1")

plt.legend()
plt.grid(True)

# Save the figure
plt.savefig("plots/comparison.png")

# Display it
plt.show()

print("Comparison graph saved as comparison.png")