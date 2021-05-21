import gym
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Discrete
from gym.spaces import Tuple
import gym_perudo
import itertools
from tqdm import tqdm

env = gym.make('perudo_game-v0')

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
                Discrete(encoder_length),
                Discrete(env.number_of_bots)))


q_table = np.zeros([env.observation_space[0].n, env.observation_space[1].n, env.observation_space[2].n, env.observation_space[3].n, env.action_space.n])

epsilon = 0.001
n_eps = 1000000
max_steps = 100
alpha = 0.85
gamma = 0.95

def choose_action(ob):
    if np.random.uniform(0,1) >= epsilon:
        a = np.argmax(q_table[ob])
    else:
        a = env.action_space.sample()
    return a

def update(ob, ob2, r, a, a2):
    predict = q_table[ob][a]
    target = r + gamma * q_table[ob2][a2]
    q_table[ob][a] = q_table[ob][a] + alpha * (target - predict)

reward=0
winner_count = []
reward_list = []
game = 0
ep_reward = []
c_reward = 0

for _ in tqdm(range(n_eps)):

    ob1 = env.reset()
    done = False
    a1 = choose_action(ob1)
    game +=1
    if env.render == True:
        print(' ')
        print('Game {}'.format(game))
    game_reward = 0

    while not done:
        ob2, r, done = env.step(a1)
        a2 = choose_action(ob2)
        update(ob1, ob2, r, a1, a2)
        ob1 = ob2
        a1 = a2
        game_reward += r

    winner_count.append(env.win)
    if env.win == 1:
        game_reward +=10
    c_reward += game_reward

    ep_reward.append(c_reward)
    final_reward = r

wins = 0
Z = []
games = 0
for win in winner_count:
    if win == 1:
        wins +=1
    Z.append(wins)
#print(winner_count)


#print('Q table \n', q_table)
X = range((n_eps))

plt.plot(X, Z)


"""def smooth_reward(ep_reward, smooth_over):
    smoothed_r = []
    for ii in range(smooth_over, len(ep_reward)):
        smoothed_r.append(np.mean(ep_reward[ii-smooth_over:ii]))
    return smoothed_r

plt.plot(smooth_reward(ep_reward, 20))
plt.title('smoothed reward')"""
plt.show()
