import numpy as np
import gym
from src import MultiArmedBandit, QLearning
import matplotlib.pyplot as plt

print('Starting example experiment')

env = gym.make('FrozenLake-v0')
epsilon1 = 0.01
epsilon2 = 0.5
steps = 100000
trials = 10
discount = 0.95

reward_matrix_q1 = np.zeros((trials, 100))
reward_matrix_q2 = np.zeros((trials, 100))

for trial in range(trials):
    agent_q1 = QLearning(epsilon=epsilon1, discount=discount)
    agent_q2 = QLearning(epsilon=epsilon2, discount=discount)
    action_values_q1, rewards_q1 = agent_q1.fit(env, steps=steps)
    action_values_q2, rewards_q2 = agent_q2.fit(env, steps=steps)

    reward_matrix_q1[trial, :] = rewards_q1
    reward_matrix_q2[trial, :] = rewards_q2

average_rewards_q1 = np.mean(reward_matrix_q1, axis=0)
average_rewards_q2 = np.mean(reward_matrix_q2, axis=0)

plt.figure()
plt.plot(range(len(average_rewards_q1)), average_rewards_q1, color='blue', label='Average rewards of QLearning for epsilon = 0.01')
plt.plot(range(len(average_rewards_q2)), average_rewards_q2, color='green', label='Average rewards of QLearning for epsilon = 0.5')
plt.title('Average Rewards of QLearning for different epsilons')
plt.xlabel('Step')
plt.ylabel('Rewards')
plt.legend(loc="best")
plt.savefig("Q3a.png")
plt.show()

print('Finished example experiment')