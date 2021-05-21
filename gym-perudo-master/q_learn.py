import gym
from gym.spaces import Discrete
from gym.spaces import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gym_perudo
import itertools
from tqdm import tqdm

env = gym.make('perudo_game-v0')

gamma = 0.95
alpha = 0.5
n_eps = 4000

epsilon_max = 1.0
epsilon_min = 0.0001
epsilon_step = (epsilon_max - epsilon_min) / n_eps

env.render = False
env.dice_per_player = 2
env.number_of_dice_sides = 2
env.number_of_bots = 1
dice_list = range(1,env.number_of_dice_sides+1)
for i in range(1,env.number_of_dice_sides+2):
    env.dict += [p for p in itertools.combinations_with_replacement(dice_list, i)]
encoder_length = len(env.dict)
env.action_space = Discrete(((env.number_of_bots+1)*env.number_of_dice_sides*env.dice_per_player)+1)
env.observation_space = Tuple((
                Discrete(((env.number_of_bots+1)*env.number_of_dice_sides*env.dice_per_player)+1),
                Discrete((env.number_of_bots+1)*env.dice_per_player+1),
                Discrete(encoder_length)))

#print('Action space: {}'.format(env.action_space.n))
#print('Observation space: {}, {}, {}'.format(env.observation_space[0].n, env.observation_space[1].n, env.observation_space[2].n))
#print('Q Table Size: {}'.format(env.action_space.n*env.observation_space[0].n*env.observation_space[1].n*env.observation_space[2].n))

q_table = np.zeros([env.observation_space[0].n, env.observation_space[1].n, env.observation_space[2].n, env.action_space.n])
epsilon = epsilon_max
game = 0
winner_count = []
reward_list = []
ep_reward = []
c_reward = 0

for _ in tqdm(range(n_eps)):

    ob = env.reset()
    done = False
    game +=1
    game_reward = 0
    if env.render == True:
        print(' ')
        print(' ')
        print('Game {}'.format(game))

    while not done:
        """if np.random.uniform(0,1) >= epsilon:"""
        a = np.argmax(q_table[ob])
        """else:
            a = env.action_space.sample()"""

        new_ob, r, done = env.step(a)
        if env.render == True:
            print('Action {}, with ob {}, and reward {}'.format(a, new_ob, r))
        q_table[ob][a] += alpha * (r + gamma * np.max(q_table[new_ob]) - q_table[ob][a])
        ob = new_ob
        game_reward += r
        epsilon = max(epsilon-epsilon_step, epsilon_min)

    winner_count.append(env.win)
    if env.win == 1:
        game_reward +=10
    c_reward += game_reward

    ep_reward.append(game_reward)
    final_reward = r
    #ep_reward.append(final_reward)


Z = []
wins = 0
games = 0
for win in winner_count:
    games +=1
    if win == 1:
        wins +=1
    Z.append(wins)

X = range((n_eps))

plt.plot(X, Z)
plt.title('Cumulative Wins')
plt.xlabel("Iterations")
plt.ylabel("Cumulative Wins")
plt.show()
"""ep_reward = [x/10000 for x in ep_reward]"""

def smooth_reward(ep_reward, smooth_over):
    smoothed_r = []
    for ii in range(smooth_over, len(ep_reward)):
        smoothed_r.append(np.mean(ep_reward[ii-smooth_over:ii]))
    return smoothed_r

plt.plot(smooth_reward(ep_reward, 20))
plt.title('Game Reward')
plt.xlabel("Iterations")
plt.ylabel("Reward per Game")
plt.show()
