import numpy as np
import matplotlib.pyplot as plt
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import your Week 3 trading environment
from week3.trading_env import ToyTradingEnv


class EpisodeRewardLogger(BaseCallback):
    """
    Records the total reward obtained in every completed episode.
    """

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self):

        # Add reward received at this timestep
        self.current_reward += self.locals["rewards"][0]

        # When an episode finishes, store the total reward
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

        return True


def train_ppo_on_trading(total_timesteps=100000):

    env = ToyTradingEnv()

    callback = EpisodeRewardLogger()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,

        # Number of environment steps collected before PPO updates the policy.
        # Larger values give more experience per update but require more memory.
        n_steps=512,

        # Number of samples used in one gradient update during training.
        # PPO splits the collected data into mini-batches of this size.
        batch_size=64,

        gamma=0.99,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    env.close()

    return model, callback.episode_rewards


def moving_average(x, window=20):

    if len(x) < window:
        return np.array(x)

    return np.convolve(
        x,
        np.ones(window) / window,
        mode="valid",
    )


def plot_rewards(
    episode_rewards,
    filename="plots/ppo_trading_rewards.png",
):

    os.makedirs("plots", exist_ok=True)

    rewards = np.array(episode_rewards)

    plt.figure(figsize=(9, 4))

    plt.plot(
        rewards,
        alpha=0.25,
        color="steelblue",
        label="Episode Reward",
    )

    plt.plot(
        moving_average(rewards),
        color="steelblue",
        linewidth=2,
        label="20-Episode Moving Average",
    )

    plt.axhline(
        0,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Zero",
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO on ToyTradingEnv")

    plt.legend()
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()

    print(f"Reward plot saved to {filename}")


if __name__ == "__main__":

    model, rewards = train_ppo_on_trading(
        total_timesteps=100000
    )

    plot_rewards(rewards)

    np.save(
        "ppo_trading_returns.npy",
        np.array(rewards),
    )

    print(f"\nFinal 20-episode average reward: {np.mean(rewards[-20:]):.2f}")

    # Compare with Week 3 random agent
    random_average = -0.0956

    print("\nComparison")
    print("--------------------------")
    print(f"Random Agent : {random_average:.2f}")
    print(f"PPO Agent    : {np.mean(rewards[-20:]):.2f}")