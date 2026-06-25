# week3_ppo.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class EpisodeRewardLogger(BaseCallback):
    """
    Records the total reward obtained in every completed episode.
    """

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:

        # Add reward obtained at this timestep
        self._current_reward += self.locals["rewards"][0]

        # When an episode finishes, save its total reward
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
            self._current_reward = 0.0

        return True


def train_ppo(env_name="CartPole-v1", total_timesteps=50000):

    env = gym.make(env_name)

    callback = EpisodeRewardLogger()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,

        # Learning rate for gradient updates
        learning_rate=3e-4,

        # PPO collects 2048 environment steps before updating
        # the neural network. Unlike REINFORCE, PPO learns
        # from batches of experience instead of one episode.
        n_steps=2048,

        # During training, the collected experience is divided
        # into mini-batches of 64 samples each.
        # Smaller batches make optimization more efficient.
        batch_size=64,

        gamma=0.99,
    )

    print("\nTraining PPO...\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    print("\nTraining Finished!\n")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=10,
    )

    print(f"PPO Evaluation: {mean_reward:.1f} +/- {std_reward:.1f}")

    # Save episode rewards for Task 3
    np.save("ppo_returns.npy", callback.episode_rewards)

    print("Saved episode rewards to ppo_returns.npy")

    env.close()

    return callback.episode_rewards


if __name__ == "__main__":

    ppo_returns = train_ppo()