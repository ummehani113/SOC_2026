# week2_reinforce.py

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """state: tensor (batch_size, state_dim) -> probs (batch_size,
        action_dim)"""
        # Raw, unbounded scores for each action ("logits"). They are not
        # probabilities yet -- they can be negative, positive, any size.
        logits = self.net(state)
        # Softmax squashes the logits into a valid probability distribution:
        # every entry is in [0, 1] and the entries sum to 1 across actions.
        # This is what makes the policy "stochastic" instead of deterministic --
        # the network expresses a belief like [0.7, 0.3] rather than picking
        # one action outright.
        probs = torch.softmax(logits, dim=-1)
        return probs


def run_episode(env, policy, render=False, gamma=0.99):
    states, actions, rewards, log_probs = [], [], [], []

    obs, info = env.reset()
    done = False

    while not done:
        # Convert the raw numpy observation from Gymnasium into a torch
        # tensor, cast to float, and add a batch dimension of size 1
        # (the network expects shape (batch_size, state_dim)).
        state = torch.from_numpy(obs).float().unsqueeze(0)

        # --- How actions are sampled from the policy ---
        # We don't need gradients here: this forward pass is only used to
        # decide which action to take while *acting* in the environment.
        # The gradient we actually care about is computed later, during
        # the learning step, using the log-probabilities we store now.
        
        probs = policy(state)

        # torch.distributions.Categorical wraps our probability vector
        # (e.g. [0.7, 0.3]) into a proper categorical distribution object
        # that knows how to sample from itself and compute log-probabilities.
        dist = torch.distributions.Categorical(probs=probs)

        # .sample() draws ONE action index at random, weighted by probs.
        # This is the source of exploration: even if action 0 currently has
        # probability 0.7, action 1 will still get chosen ~30% of the time,
        # so the agent keeps trying alternatives instead of always picking
        # whichever action currently looks best.
        action = dist.sample()

        if render:
            env.render()

        # Step the environment forward using the sampled action.
        # action.item() converts the 1-element tensor into a plain Python
        # int, since Gymnasium expects a plain integer action.
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        # An episode ends either because the task naturally finished
        # (terminated, e.g. the pole fell) or because we hit a time limit
        # (truncated, e.g. max steps reached). Either way, done = True.
        done = terminated or truncated

        # Record everything we'll need later to compute returns and the
        # loss: the state we were in, the action we took, the reward we
        # got, and the log-probability of that action under the policy
        # at the time we took it.
        states.append(obs)
        actions.append(action.item())
        rewards.append(reward)
        # dist.log_prob(action) gives log(pi_theta(action | state)) --
        # exactly the log-probability term needed for the policy gradient.
        # We store it now (with gradient tracking, since this call is
        # NOT inside torch.no_grad()) so we can backpropagate through it
        # after the episode ends.
        log_probs.append(dist.log_prob(action))

        obs = next_obs

    # --- How returns are computed ---
    # The return G_t at time t is the (discounted) sum of all rewards from
    # t until the end of the episode: G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
    #
    # Computing this efficiently is done backwards through the episode:
    # start from the LAST reward and work toward the first, maintaining a
    # running total G. At each step we do G = r + gamma * G, then store G
    # as the return for that time step. Going backwards lets each G_t
    # reuse the already-computed G_{t+1} instead of re-summing a long list
    # of future rewards from scratch for every single time step.
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        # Insert at the front because we are filling the list from the
        # end of the episode back to the start, but we want returns[t]
        # to line up with rewards[t] (chronological order).
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    # Stack the list of individual log-prob tensors (one per time step)
    # into a single 1-D tensor so it can be combined with `returns` in
    # one vectorized operation during the loss computation.
    log_probs = torch.stack(log_probs)

    return states, actions, rewards, returns, log_probs


def train_reinforce(
    env_name="CartPole-v1",
    hidden_dim=64,
    learning_rate=1e-2,
    gamma=0.99,
    num_episodes=500,
):
    env = gym.make(env_name, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    all_episode_returns = []

    for episode in range(num_episodes):
        states, actions, rewards, returns, log_probs = run_episode(
            env, policy, render=False, gamma=gamma
        )

        # --- Variance reduction via normalization (acts as a baseline) ---
        # Raw returns can vary wildly in scale across episodes just due to
        # randomness, which makes gradient estimates noisy. Subtracting the
        # batch mean and dividing by the batch std turns "raw return" into
        # "how much better/worse than a typical episode was this time step",
        # which is a much more stable learning signal. This does not change
        # the expected direction of the gradient, only its variance.
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8  # avoid division by zero
        normalized_returns = (returns - returns_mean) / returns_std

        # --- How the REINFORCE loss is formed, and why it has a minus sign ---
        # The policy gradient theorem tells us to adjust theta in the
        # direction that INCREASES:
        #     sum_t [ log pi_theta(a_t | s_t) * G_t ]
        # i.e. we want to perform gradient ASCENT on this quantity (push
        # up the log-probability of actions that led to good returns, and
        # push it down for actions that led to bad returns).
        #
        # PyTorch's optimizers (like Adam here) are built to perform
        # gradient DESCENT -- they minimize whatever scalar you call
        # .backward() on. To reuse that machinery for what is conceptually
        # an ascent problem, we simply negate the quantity we want to
        # maximize and minimize that instead. Minimizing -X is mathematically
        # equivalent to maximizing X, so:
        #     loss = -sum_t [ log pi_theta(a_t | s_t) * G_t ]
        # This is exactly why the minus sign appears below: it converts our
        # "maximize expected return" goal into a "minimize this loss" problem
        # that optimizer.step() knows how to handle.
        loss = -(log_probs * normalized_returns).sum()

        # Standard PyTorch training step:
        # 1) clear old gradients, 2) backpropagate the new loss to compute
        # fresh gradients w.r.t. policy parameters, 3) take an optimizer
        # step that nudges parameters in the direction that reduces the loss
        # (equivalently, increases expected return).
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the (undiscounted) total reward for this episode, purely
        # for logging/plotting -- this is NOT used in the loss itself.
        episode_return = sum(rewards)
        all_episode_returns.append(episode_return)

        if (episode + 1) % 10 == 0:
            avg_last_10 = np.mean(all_episode_returns[-10:])
            print(
                f"Episode {episode + 1:4d} | "
                f"Return: {episode_return:6.1f} | "
                f"Avg last 10: {avg_last_10:6.1f}"
            )

    env.close()
    return all_episode_returns, policy


if __name__ == "__main__":

    experiments = [
        ("Baseline", 1e-2, 0.99),
        ("High Learning Rate", 1e-1, 0.99),
        ("Low Learning Rate", 1e-5, 0.99),
        ("Low Gamma", 1e-2, 0.50),
    ]

    results = {}

    for name, lr, gamma in experiments:

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        returns, _ = train_reinforce(
            learning_rate=lr,
            gamma=gamma,
            num_episodes=300,
        )

        results[name] = returns

        print(f"\nLearning Rate : {lr}")
        print(f"Gamma         : {gamma}")
        print(f"Average Return (last 10 episodes): {np.mean(returns[-10:]):.2f}")

    # ---------------- Plot all experiments ----------------

    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(10,6))

    # for name, returns in results.items():
    #     plt.plot(returns, label=name)

    # plt.xlabel("Episode")
    # plt.ylabel("Episode Return")
    # plt.title("REINFORCE Hyperparameter Comparison")
    # plt.legend()
    # plt.grid(True)

    # plt.show()