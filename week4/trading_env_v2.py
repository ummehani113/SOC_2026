# week4_trading_env_v2.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from collections import deque
class TradingEnvV2(gym.Env):
    """
    Extended trading environment with:
    - A window of the last 'history_len ' price returns in the state.
    - A transaction cost subtracted from the reward when position
    changes.
    Observation (history_len + 1 dimensional):
    [r_{t-history_len +1}, ..., r_t , current_position]
    Actions (discrete):
    0 = Go Long
    1 = Hold
    2 = Go Short
    Reward:
    position * price_return * 100
    - transaction_cost * |new_position - old_position|
    """
    def __init__(
        self,
        episode_length=100,
        initial_price=100.0,
        daily_vol=0.01,
        history_len=5,
        transaction_cost=0.1,
    ):
        super().__init__()
        self.episode_length = episode_length
        self.initial_price = initial_price
        self.daily_vol = daily_vol
        self.history_len = history_len
        self.transaction_cost = transaction_cost
        obs_dim = history_len + 1 # history returns + current position
        self.observation_space = spaces.Box(
            low=np.full(obs_dim , -np.inf , dtype=np.float32),
            high=np.full(obs_dim , np.inf , dtype=np.float32),
        )
        self.action_space = spaces.Discrete (3)
        self.price = None
        self.position = None
        self.step_count = None
        self._price_history = None
# ---------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.price = self.initial_price
        self.position = 0.0
        self.step_count = 0
        self._price_history = deque(
            [0.0] * self.history_len, maxlen=self.history_len
        )
        return self._get_obs(last_return=0.0), {}
# ---------------------------------------------------------------- #
    def step(self, action):
        # Price dynamics
        price_return = float(
            self.np_random.normal(0.0, self.daily_vol)
        )
        self.price *= (1.0 + price_return)
        self._price_history.append(price_return)
        # Map action to new position
        action_to_position = {0: 1.0, 1: self.position, 2: -1.0}
        new_position = action_to_position[int(action)]
        # Reward: P&L from old position minus transaction cost
        pnl = self.position * price_return * 100.0
        cost = self.transaction_cost * abs(new_position - self.position)
        reward = pnl - cost
        # TODO: add a comment here explaining why cost uses
        # (new_position - self.position) and when it equals zero
        self.position = new_position
        self.step_count += 1
        terminated = self.step_count >= self.episode_length
        truncated = False
        return self._get_obs(price_return), reward, terminated, truncated, {}
# ---------------------------------------------------------------- #
    def _get_obs(self, last_return):
        history = np.array(list(self._price_history), dtype=np.float32)
        return np.append(history, self.position).astype(np.float32)
# ---------------------------------------------------------------- #
    def render(self):
        print(
            f"Step {self.step_count :3d} | "
            f"Price {self.price :8.2f} | "
            f"Position {self.position :+.0f}"
        )

from gymnasium.utils.env_checker import check_env
from trading_env_v2 import TradingEnvV2

check_env(TradingEnvV2(), warn=True)

print("Environment check passed!")

env = TradingEnvV2()

for episode in range(3):

    obs, info = env.reset()

    done = False
    total_reward = 0

    while not done:

        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

env.close()

from week3.trading_env import ToyTradingEnv

env1 = ToyTradingEnv()
env2 = TradingEnvV2()

print("\nObservation Shapes")
print("------------------")
print("ToyTradingEnv :", env1.observation_space.shape)
print("TradingEnvV2  :", env2.observation_space.shape)

env1.close()
env2.close()