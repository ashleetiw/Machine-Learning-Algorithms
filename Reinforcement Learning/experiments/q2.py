import numpy as np
import gym
from src import MultiArmedBandit, QLearning
import matplotlib.pyplot as plt

print('Starting example experiment')

env = gym.make('SlotMachines-v0')
steps = 100000
trials = 10
action_matrix_bandits = np.zeros((trials, 10))
reward_matrix_bandits = np.zeros((trials, 100))
action_matrix_q = np.zeros((trials, 10))
reward_matrix_q = np.zeros((trials, 100))

for trial in range(trials):
    agent_bandits = MultiArmedBandit()
    agent_q = QLearning()
    action_values_bandits, rewards_bandits = agent_bandits.fit(env, steps=steps)
    action_values_q, rewards_q = agent_q.fit(env, steps=steps)

    action_matrix_bandits[trial, :] = action_values_bandits
    reward_matrix_bandits[trial, :] = rewards_bandits
    action_matrix_q[trial, :] = action_values_q
    reward_matrix_q[trial, :] = rewards_q

average_5_rewards_bandits = np.mean(reward_matrix_bandits[:5], axis=0)
average_10_rewards_bandits = np.mean(reward_matrix_bandits, axis=0)
average_10_rewards_q = np.mean(reward_matrix_q, axis=0)

plt.figure()
plt.plot(range(len(reward_matrix_bandits[0])), reward_matrix_bandits[0], color='orange', label='Rewards from the first trial')
plt.plot(range(len(average_5_rewards_bandits)), average_5_rewards_bandits, color='blue', label='Average rewards over 5 trials')
plt.plot(range(len(average_10_rewards_bandits)), average_10_rewards_bandits, color='green', label='Average rewards over all trials')
plt.title('Rewards of MultiArmedBandit')
plt.xlabel('Step')
plt.ylabel('Rewards')
plt.legend(loc="best")
plt.savefig("Q2a.png")
plt.show()

plt.figure()
plt.plot(range(len(average_10_rewards_bandits)), average_10_rewards_bandits, color='orange', label='Average rewards of MultiArmedBandit')
plt.plot(range(len(average_10_rewards_q)), average_10_rewards_q, color='green', label='Average rewards of QLearning')
plt.title('Average Rewards of QLearning & MultiArmedBandit')
plt.xlabel('Step')
plt.ylabel('Rewards')
plt.legend(loc="best")
plt.savefig("Q2b.png")
plt.show()

print('Finished example experiment')