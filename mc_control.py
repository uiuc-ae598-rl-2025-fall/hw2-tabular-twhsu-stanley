import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from agent import Agent

"""
def epsilon_greedy_policy(Q, state, n_action, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(n_action)
    else:
        return np.argmax(Q[state, :])

def simulate_episode(env, policy):
    state, info = env.reset() # starting state at 0
    state_hist = [state]
    action_hist = []
    reward_hist = []
    while True:
        action = policy[state]
        action_hist.append(action)

        state, reward, done, truncated, info = env.step(action)
        state_hist.append(state)
        reward_hist.append(reward)
        
        if state == 15: # 15 is the goal state
            reach_goal = True
            break
        if done or truncated:
            break
    return state_hist, action_hist, reward_hist

def check_first_visit(state_hist, action_hist, t):
    for i in range(t):
        if state_hist[i] == state_hist[t] and action_hist[i] == action_hist[t]:
            return False
    return True

def mc_control(env, gamma, learning_rate, epsilon, n_espisodes):

    n_state = env.observation_space.n
    n_action = env.action_space.n

    # Initialize Q
    Q = np.zeros((n_state, n_action))

    # Initialize policy
    pi = np.zeros(n_state, dtype=int)
    for s in range(n_state):
        pi[s] = epsilon_greedy_policy(Q, s, n_action, epsilon)
    
    for episode in range(n_espisodes):
        state_hist, action_hist, reward_hist = simulate_episode(env, pi)

        # epsilon decay to impose GLIE
        epsilon_d = epsilon / (episode + 1)

        G = 0
        for t in (len(state_hist) - range(state_hist) - 2):
            G = gamma * G + reward_hist[t+1]

            # check first visit
            if check_first_visit(state_hist, action_hist, t):
                continue

            Q[state_hist[t], action_hist[t]] += learning_rate * (G - Q[state_hist[t], action_hist[t]])

            pi[state_hist[t]] = epsilon_greedy_policy(Q, state_hist[t], n_action, epsilon_d)
"""

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