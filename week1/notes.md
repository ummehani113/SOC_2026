1. Observation = [cart position, cart velocity, pole angle, pole angular velocity]

2. Why does the pole fall so quickly?

The pole falls quickly because the agent is taking random actions using env.action_space.sample(). It has not learned how to balance the pole, so it randomly pushes the cart left or right. Most random actions do not move the cart underneath the falling pole, causing the pole's angle to increase until it exceeds the allowed limit (±12°), ending the episode. Gravity continuously pulls the pole downward, so without an intelligent policy, it falls within a few steps.

3. The reward is +1 per step the pole stays up. What's the maximum total reward possible?

For CartPole-v1, the agent receives +1 reward for every timestep, including the final timestep before the episode ends. An episode is automatically truncated after 500 steps, so the maximum possible total reward is 500. This represents a perfect episode where the agent keeps the pole balanced for the entire allowed duration.

# Part 2: Reflection

4. What rule did you use?

I used a simple hand-crafted policy based on the pole's angle. If the pole was leaning to the right (positive pole angle), I moved the cart to the right. If the pole was leaning to the left (negative pole angle), I moved the cart to the left. The idea was to keep the cart underneath the falling pole, similar to balancing a broomstick on your hand.

5. What was your average score?

My average score was 45.8.

6. Did the rule work every time, or did it sometimes fail? Why?

The rule performed much better than taking random actions, but it did not work perfectly every time. It only considered the current pole angle and ignored other important information such as the pole's angular velocity, the cart's position, and the cart's velocity. As a result, it often reacted too late or made decisions that were not optimal, causing the pole to eventually fall.

7. Can you think of a smarter rule?

A smarter rule would consider both the pole's angle and its angular velocity. For example, if the pole is already rotating quickly toward one side, the cart should start moving in that direction even before the pole leans too far. An even better approach would also use the cart's position and velocity to make more informed decisions. Instead of manually designing these rules, a Reinforcement Learning agent can learn an effective policy automatically through trial and error.

# Part 3: Trying Different Reinforcement Learning Environments

8. Which environments did you try?

I explored all three environments:

MountainCar-v0
Acrobot-v1
unarLander-v2


9. What is the agent trying to do?

 MountainCar-v0
The agent controls a car that is stuck in a valley between two hills. The car's engine is too weak to drive directly to the top of the right hill. The objective is to build momentum by first driving back and forth, then using that momentum to reach the goal at the top of the hill.

 Acrobot-v1
The agent controls a two-link robotic arm that starts hanging downward. The objective is to swing the arm back and forth until the free end reaches a target height above the ground, similar to pumping a swing higher and higher.

 LunarLander-v2
The agent controls a spacecraft that must land safely on a landing pad. It has to reduce its speed, maintain a stable orientation, and touch down gently without crashing or drifting away.


10. What does a state look like? What are the possible actions?

 MountainCar-v0
 **State:** 2 values
  * Car position
  * Car velocity
* **Actions:**
  * 0 → Push left
  * 1 → No acceleration
  * 2 → Push right

 Acrobot-v1
* **State:** 6 values describing the joint angles and angular velocities of the two links.
* **Actions:**
  * 0 → Apply torque left
  * 1 → No torque
  * 2 → Apply torque right

 LunarLander-v2
 **State:** 8 values including the lander's position, velocity, angle, angular velocity, and whether each leg is touching the ground.
* **Actions:**
  * 0 → Do nothing
  * 1 → Fire left engine
  * 2 → Fire main engine
  * 3 → Fire right engine



11. Does random play ever solve it? Why or why not?

MountainCar-v0

Random actions almost never solve the task. Reaching the goal requires a specific strategy of moving back and forth to build momentum, which random actions rarely achieve.

Acrobot-v1

Random actions rarely succeed because the robot must swing in a coordinated way to gain enough energy. Random torques do not generate the required motion.

LunarLander-v2

Random actions almost always result in a crash. Safe landing requires precise control of thrust, direction, speed, and orientation, which cannot be achieved consistently through random actions.

# Part 4: Big-Picture Reflection

12. In your own words, what is the agent's goal in any RL problem?

The agent's goal is to choose actions that maximize the total reward it receives over time. It interacts with the environment, learns from the rewards, and tries to find the best way to complete the given task.


13. Why is "random" not a good strategy — and what would the agent need to do instead?

Random actions do not consider the current state of the environment, so they usually lead to poor performance. Instead, the agent should learn which actions are better in different situations and use that knowledge to make smarter decisions.


14. What does it mean for an agent to "learn"? (Hint: what should change between episode 1 and episode 100?)

Learning means improving its decision-making based on past experience. By episode 100, the agent should perform better than it did in episode 1 because it has learned which actions lead to higher rewards and which ones should be avoided.


15. Write down 3 questions you have about RL that you'd like to ask your mentor.

1. How does an RL agent actually learn from rewards without being told the correct action?
2. How does the agent balance exploring new actions and using actions that already give good rewards?
3. How are reward functions designed for real-world problems like robotics or self-driving cars?



