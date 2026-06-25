Explain in your own words what REINFORCE does with one episode of data — basically: "it plays an episode, sees how well it went, then makes good actions more likely and bad actions less likely."
Draw a simple loop diagram:
Collect episodes → Compute returns → Compute loss → Update the network (optimizer step) → repeat
Read a learning curve (a graph of reward over time) and tell whether the agent is getting better, has plateaued, or is stuck/not learning at all.

Task 3- Reflections:
1. The episode returns generally increased as training progressed, showing that the agent was learning to balance the pole better. The moving average also increased over time, although there were small fluctuations because the agent still explored different actions.

2. The trained policy balanced the pole for much longer than the random policy. The random policy failed quickly, while the trained policy made more stable and purposeful decisions to keep the pole upright.

3. I increased the number of training episodes from 300 to 500. The agent learned more consistently, and the average return improved compared to the shorter training run.

4. The REINFORCE loss tries to increase the probability of actions that resulted in higher returns and decrease the probability of actions that performed poorly. This helps the policy gradually learn better actions through experience.
