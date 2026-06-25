import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()

print(f"Initial observation: {obs}")
print(f"Observation = [ship position, ship velocity, ship angle, angular velocity]")
print(f"Action space: {env.action_space}  (0 = do nothing, 1 = fire left engine, 2 = fire main engine, 3 = fire right engine)")

total_reward = 0

steps = 0

done = False
while not done:
    action = env.action_space.sample()  # pick a random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    done = terminated or truncated

print(f"\nEpisode ended after {steps} steps with total reward {total_reward}")
env.close()