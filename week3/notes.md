1. Which hyperparameter change hurt learning the most? What does this tell you about gradient-based optimisation?

Increasing the learning rate too much hurt learning the most. The rewards became unstable and the agent stopped improving consistently. This showed me that gradient-based optimisation is very sensitive to hyperparameters, and even a small change can affect training a lot.

2. REINFORCE updates after every single episode. Why does using one trajectory’s data make the gradient estimate noisy?

Each episode is different because the agent explores different actions. Sometimes it gets lucky and sometimes it doesn't, so the update is based on only one experience. This makes the gradient noisy and causes the learning curve to fluctuate.

3. PPO uses a clipped probability ratio. In your own words, what problem does the clip prevent?

The clipping prevents the policy from changing too much after a single update. It keeps learning stable by making sure the agent improves gradually instead of suddenly changing its behaviour. This helps avoid unstable training.

4. PPO trains on mini-batches from a large replay buffer. Why might this produce a smoother reward curve than REINFORCE?

PPO learns from a larger collection of experiences instead of only one episode. Since the updates are based on more data, the learning signal is less noisy. This usually leads to a smoother and more stable reward curve.

5. Describe the two reward curves in 3–4 sentences. Which learned faster? Which was more consistent? Were there any surprises?

The PPO reward curve increased steadily and quickly reached the maximum reward of 500. The REINFORCE curve improved very slowly and stayed almost flat throughout training. PPO clearly learned faster and was much more consistent. I was surprised by how large the performance difference was between the two methods.

6. List 5 features you would add to the state to make the environment more realistic. For each, explain why it would help the agent.

* Moving average of price – helps identify trends.
* Trading volume – shows how active the market is.
* Volatility – helps the agent estimate market risk.
* Previous few returns – provides short-term market history.
* Cash balance or portfolio value – helps the agent make decisions based on available capital instead of only price.

7. The current reward is position * price_return * 100. What happens if the agent always stays long? Is that a good strategy on a random walk?

If the agent always stays long, it earns money only when prices go up and loses money when prices go down. Since the prices follow a random walk, gains and losses roughly cancel out over time. So always staying long is not a reliable strategy.

8. What are three limitations of this toy environment compared to a real financial market? How might each limitation affect what the agent learns?

The environment has completely random prices, so there are no real patterns to learn. It also ignores transaction costs, making frequent trading unrealistically cheap. Finally, it only tracks one asset with a very simple state, so the agent cannot learn more realistic market behaviour.

9. What was the mean reward of the random agent over 10 episodes? Was it positive, negative, or near zero? Does that match your expectation?

The mean reward over 10 episodes was approximately -0.096, which is very close to zero. It was slightly negative but essentially near zero overall. This matches my expectation because the price movements are random and the agent is also taking random actions.

10. Write down 3 questions about environment design or RL-for-trading that you want to discuss with your mentor.

* How are realistic stock market environments created for reinforcement learning?
* How do professional trading agents balance exploration and exploitation without taking excessive risk?
* What changes are needed to make a toy trading environment useful for solving real-world trading problems?
