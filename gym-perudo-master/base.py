import gym
from gym.spaces import Discrete
from gym.spaces import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import gym_perudo
import itertools
from tqdm import tqdm
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('perudo_game-v0')

gamma = 0.95
alpha = 0.05
n_eps = 100000

epsilon_max = 1.0
epsilon_min = 0.001
epsilon_decay = 0.99999
epsilon_step = (epsilon_max - epsilon_min) / n_eps


env.render = True
env.dice_per_player = 3
env.number_of_dice_sides = 2
env.number_of_bots = 2
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

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
