# =============================================================================
# IMPORTING THE GYMNASIUM LIBRARY
# =============================================================================

# Gymnasium is a Python library that provides ready-made Reinforcement Learning
# environments like CartPole, MountainCar, LunarLander, etc.
#
# Instead of writing the physics, rewards, gravity, collisions ourselves,
# Gymnasium already implements everything.
#
# "as gym" is simply an alias (nickname), so instead of writing
# gymnasium.make(...)
# we can write
# gym.make(...)

import gymnasium as gym


# =============================================================================
# POLICY FUNCTION
# =============================================================================
#
# In Reinforcement Learning, a POLICY is simply a rule that tells the agent:
#
#       Current State  --->  Which Action should I take?
#
# Here, we are NOT using Machine Learning.
# We are manually writing our own decision rule.

def my_policy(observation):

    # observation contains the COMPLETE STATE of the environment.
    #
    # observation =
    # [
    #     cart_position,
    #     cart_velocity,
    #     pole_angle,
    #     pole_angular_velocity
    # ]
    #
    # Example:
    # observation = [0.03, 0.21, -0.04, 0.56]

    # "_" means "ignore this value".
    # We only care about pole_angle for this simple policy.

    _, _, pole_angle, _ = observation

    # CartPole has only TWO possible actions:
    #
    # 0 = Push cart LEFT
    # 1 = Push cart RIGHT
    #
    # Logic:
    #
    # If the pole is leaning RIGHT (positive angle),
    # move the cart RIGHT.
    #
    # If the pole is leaning LEFT (negative angle),
    # move the cart LEFT.
    #
    # This is similar to balancing a broomstick on your hand.

    return 1 if pole_angle > 0 else 0


# =============================================================================
# CREATE THE ENVIRONMENT
# =============================================================================
#
# gym.make() creates a new RL environment.
#
# "CartPole-v1" tells Gymnasium which game/environment to load.
#
# render_mode="human"
# opens a graphical window so that we can WATCH the simulation.

env = gym.make("CartPole-v1", render_mode="human")


# =============================================================================
# STORE REWARDS OF ALL EPISODES
# =============================================================================
#
# We will run multiple episodes.
# The total reward from each episode will be stored in this list.

scores = []


# =============================================================================
# RUN 5 EPISODES
# =============================================================================
#
# One episode means:
#
# Start with a fresh environment
# ↓
# Balance the pole until it falls
# ↓
# Episode ends
#
# Repeat this process 5 times.

for episode in range(5):

    # -------------------------------------------------------------------------
    # RESET THE ENVIRONMENT
    # -------------------------------------------------------------------------
    #
    # Every episode starts with:
    #
    # Cart at center
    # Pole almost vertical
    #
    # reset() returns:
    #
    # obs  -> initial observation (state)
    # info -> extra information (not needed here)

    obs, info = env.reset()

    # Total reward earned in this episode
    total_reward = 0

    # Episode is not finished initially
    done = False


    # -------------------------------------------------------------------------
    # MAIN RL LOOP
    # -------------------------------------------------------------------------
    #
    # Every Reinforcement Learning algorithm follows this loop:
    #
    # Observe State
    # ↓
    # Choose Action
    # ↓
    # Environment executes action
    # ↓
    # Receive Reward + New State
    # ↓
    # Repeat

    while not done:

        # -------------------------------------------------------------
        # STEP 1 : Agent chooses an action using its policy
        # -------------------------------------------------------------

        action = my_policy(obs)


        # -------------------------------------------------------------
        # STEP 2 : Environment executes the action
        # -------------------------------------------------------------
        #
        # env.step(action) returns FIVE things:
        #
        # obs          -> new observation (new state)
        #
        # reward       -> reward received after taking the action
        #
        # terminated   -> True if pole fell or cart crossed limits
        #
        # truncated    -> True if maximum time limit (500 steps) reached
        #
        # info         -> extra information (usually ignored)

        obs, reward, terminated, truncated, info = env.step(action)


        # -------------------------------------------------------------
        # STEP 3 : Update total reward
        # -------------------------------------------------------------
        #
        # CartPole gives:
        #
        # +1 reward for every timestep the pole remains balanced.
        #
        # Example:
        #
        # Step 1 -> +1
        # Step 2 -> +1
        # Step 3 -> +1
        #
        # Total reward keeps increasing.

        total_reward += reward


        # -------------------------------------------------------------
        # STEP 4 : Check if episode has finished
        # -------------------------------------------------------------
        #
        # Episode ends if:
        #
        # 1. Pole falls
        # OR
        # 2. 500 steps completed

        done = terminated or truncated


    # -------------------------------------------------------------------------
    # Episode finished
    # -------------------------------------------------------------------------

    # Save reward of this episode
    scores.append(total_reward)

    print(f"Episode {episode+1}: {total_reward}")


# =============================================================================
# CALCULATE AVERAGE PERFORMANCE
# =============================================================================
#
# Suppose scores =
#
# [55, 70, 63, 81, 59]
#
# Average =
#
# (55 + 70 + 63 + 81 + 59) / 5

average_reward = sum(scores) / len(scores)

print(f"\nAverage Reward: {average_reward:.1f}")


# =============================================================================
# CLOSE THE ENVIRONMENT
# =============================================================================
#
# Closes the simulation window and frees resources.

env.close()