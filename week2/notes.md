Explain in your own words what REINFORCE does with one episode of data — basically: "it plays an episode, sees how well it went, then makes good actions more likely and bad actions less likely."
Draw a simple loop diagram:
Collect episodes → Compute returns → Compute loss → Update the network (optimizer step) → repeat
Read a learning curve (a graph of reward over time) and tell whether the agent is getting better, has plateaued, or is stuck/not learning at all.

Task 3- Reflections:
1. The episode returns generally increased as training progressed, showing that the agent was learning to balance the pole better. The moving average also increased over time, although there were small fluctuations because the agent still explored different actions.

2. The trained policy balanced the pole for much longer than the random policy. The random policy failed quickly, while the trained policy made more stable and purposeful decisions to keep the pole upright.

3. I increased the number of training episodes from 300 to 500. The agent learned more consistently, and the average return improved compared to the shorter training run.

4. The REINFORCE loss tries to increase the probability of actions that resulted in higher returns and decrease the probability of actions that performed poorly. This helps the policy gradually learn better actions through experience.

MY NOTES:
 What is a Policy?

A policy is simply

a function that decides which action to take for every state.

Notation:

π(a∣s)

means

Probability of taking action a

given state s.

Instead of saying

Always push right

our policy says

Push Left  : 30%

Push Right : 70%

The action is then randomly sampled according to these probabilities.

This randomness helps exploration.


5. Parameterized Policy

Instead of a fixed rule,

we use a neural network.

Input

4 observations

↓

Neural Network

↓

Output

Probability(Left)

Probability(Right)

The parameters (weights) inside this neural network are represented by

θ

These parameters are what we train.


6. Neural Network Architecture

Our network was

Input Layer
4 neurons

↓

Hidden Layer
64 neurons

↓

Output Layer
2 neurons

Implemented as

nn.Linear(4,64)

↓

ReLU

↓

nn.Linear(64,2)

↓
Softmax

7. Sampling an Action

Unlike supervised learning,

we do not always pick the action with the highest probability.

Instead,

we create a probability distribution.

dist = torch.distributions.Categorical(probs)

Then

action = dist.sample()

Suppose

Left  = 0.2

Right = 0.8

Then

Right will usually be chosen,

but Left still has a 20% chance.

This helps the agent explore.






By the end of Week 2, i understood:

What a policy is.
How a neural network can represent a policy.
Why action probabilities are used instead of fixed actions.
How actions are sampled from a probability distribution.
How rewards are converted into returns.
How REINFORCE updates the policy based on episode outcomes.
How backpropagation trains a policy network in reinforcement learning.
How to train a policy gradient agent on CartPole and evaluate its learning progress.