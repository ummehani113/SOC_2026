# week3_trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ToyTradingEnv(gym.Env):
    """
    A simple trading environment.

    Observation:
        [last_price_return, current_position]

    Position:
        -1 = Short
         0 = Flat
         1 = Long

    Actions:
        0 = Go Long
        1 = Hold current position
        2 = Go Short
    """

    def __init__(self, episode_length=100, initial_price=100.0, daily_vol=0.01):
        super().__init__()

        # Number of steps in one episode
        self.episode_length = episode_length

        # Starting asset price
        self.initial_price = initial_price

        # Standard deviation of daily returns
        self.daily_vol = daily_vol

        # Observation = [price_return, position]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -1.0], dtype=np.float32),
            high=np.array([np.inf, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 3 discrete actions:
        # 0 = Long
        # 1 = Hold
        # 2 = Short
        self.action_space = spaces.Discrete(3)

        self.price = None
        self.position = None
        self.step_count = None

    def reset(self, seed=None, options=None):
        """
        Starts a new episode.
        """

        super().reset(seed=seed)

        self.price = self.initial_price
        self.position = 0.0          # Start with no position
        self.step_count = 0

        observation = np.array(
            [0.0, self.position],
            dtype=np.float32
        )

        info = {}

        return observation, info

    def step(self, action):
        """
        Executes one environment step.
        """

        # Generate a random daily return.
        # self.np_random comes from Gymnasium and respects the environment seed.
        price_return = float(
            self.np_random.normal(0.0, self.daily_vol)
        )

        # Update the asset price
        self.price *= (1.0 + price_return)

        # Convert action into a trading position.
        action_to_position = {
            0: 1.0,               # Long
            1: self.position,     # Hold
            2: -1.0               # Short
        }

        new_position = action_to_position[int(action)]

        # Reward is based on the OLD position.
        # If we were long and price increased,
        # reward is positive.
        reward = self.position * price_return * 100.0

        # Update to the new position
        self.position = new_position

        self.step_count += 1

        terminated = self.step_count >= self.episode_length
        truncated = False

        observation = np.array(
            [price_return, self.position],
            dtype=np.float32
        )

        info = {}

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Prints the current environment state.
        """

        print(
            f"Step {self.step_count:3d} | "
            f"Price {self.price:8.2f} | "
            f"Position {self.position:+.0f}"
        )


def run_random_agent(n_episodes=10):
    """
    Runs a random policy and reports total reward.
    """

    env = ToyTradingEnv()

    total_rewards = []

    for episode in range(n_episodes):

        observation, info = env.reset()

        total_reward = 0.0
        done = False

        while not done:

            # Randomly choose Buy/Hold/Sell
            action = env.action_space.sample()

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            done = terminated or truncated

        total_rewards.append(total_reward)

        print(
            f"Episode {episode+1:2d}: "
            f"Total Reward = {total_reward:.2f}"
        )

    print("\nMean Reward:", np.mean(total_rewards))

    env.close()

    return total_rewards


if __name__ == "__main__":

    # Run a random agent for 10 episodes
    run_random_agent()

    # Verify that the environment follows Gymnasium's API correctly
    from gymnasium.utils.env_checker import check_env

    check_env(ToyTradingEnv(), warn=True)

    print("\nEnvironment check passed!")