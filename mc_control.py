import numpy as np
import gymnasium as gym
from agent import Agent

"""On-policy first-visit MC control"""

if __name__ == "__main__":
    # Create FrozenLake environment
    # State: 0 to 15
    # Action: 0: Left, 1: Down, 2: Right, 3: Up
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=["SFFF", "FHFH", "FFFH", "HFFG"], map_name="4x4", is_slippery=is_slippery)

    gamma = 0.95
    n_espisodes = 20000
    epsilon = 1.0
    learning_rate = 0.2

    mc_agent = Agent(env, gamma, learning_rate, epsilon)
    mc_agent.mc_control(n_espisodes)
    mc_agent.plot_value_and_policy(is_slippery=is_slippery, algorithm="MC Control")