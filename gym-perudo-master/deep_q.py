import gym
from gym.spaces import Discrete
from gym.spaces import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import gym_perudo
import itertools
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import random

env = gym.make('perudo_game-v0')
env.reset()

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


class DQNAgent():
    def __init__(self, env, epsilon=1.0, alpha=0.5, gamma=0.9, time = 30000):
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = env.observation_space
        self.memory = deque(maxlen=2000) # Record past experiences- [(state, action, reward, next_state, done)...]
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        self.gamma = gamma       # Discount factor- closer to 1 learns well into distant future
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning = True
        self.model = self._build_model()

        self.time = time
        self.time_left = time # Epsilon Decay
        self.small_decrement = (0.4 * epsilon) / (0.3 * self.time_left) # reduce epsilon

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_shape = (2,), kernel_initializer='random_uniform', activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.alpha))

        return model

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:

            action = random.randrange(self.action_size)


        else:

            action_value = self.model.predict(state)
            action = np.argmax(action_value[0])

        self.update_parameters()
        return action

    def update_parameters(self):
        if self.time_left > 0.9 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.7 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.5 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.3 * self.time:
            self.epsilon -= self.small_decrement
        elif self.time_left > 0.1 * self.time:
            self.epsilon -= self.small_decrement
        self.time_left -= 1


    def learn(self, state, action, reward, next_state, done):
        target = reward

        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        target_f = self.model.predict(state)


        target_f[0][action] = target

        self.model.fit(state, target_f, epochs=1, verbose=0)


#             print(self.time)
#         print(self.epsilon)

    def get_optimal_strategy(self):
        index = []
        for x in range(0,21):
            for y in range(1,11):
                index.append((x,y))

        df = pd.DataFrame(index = index, columns = ['Stand', 'Hit'])

        for ind in index:
            outcome = self.model.predict([np.array([ind])], batch_size=1)
            df.loc[ind, 'Stand'] = outcome[0][0]
            df.loc[ind, 'Hit'] = outcome[0][1]


        df['Optimal'] = df.apply(lambda x : 'Hit' if x['Hit'] >= x['Stand'] else 'Stand', axis=1)
        df.to_csv('optimal_policy.csv')
        return df

start_time = time.time()

# print(agent.model)

num_rounds = 100 # Payout calculated over num_rounds
num_samples = 50 # num_rounds simulated over num_samples

agent = DQNAgent(env=env, epsilon=1.0, alpha=0.001, gamma=0.1, time=7500)

average_payouts = []

state = env.reset()
state = np.reshape(state[0:2], [1,2])
for sample in range(num_samples):
    round = 1
    total_payout = 0 # store total payout per sample
    while round <= num_rounds:
        action = agent.choose_action(state)
        next_state, payout, done = env.step(action)
        next_state = np.reshape(next_state[0:2], [1,2])


        total_payout += payout
#         if agent.learning:
        agent.learn(state, action, payout, next_state, done)

        state = next_state
        state = np.reshape(state[0:2], [1,2])

        if done:
            state = env.reset() # Environment deals new cards to player and dealer
            state = np.reshape(state[0:2], [1,2])
            round += 1

    average_payouts.append(total_payout)

    if sample % 10 == 0:
        print('Done with sample: ' + str(sample) + str("   --- %s seconds ---" % (time.time() - start_time)))
        print(agent.epsilon)

print(agent.get_optimal_strategy())

# Plot payout per 1000 episodes for each value of 'sample'

plt.plot(average_payouts)
plt.xlabel('num_samples')
plt.ylabel('payout after 1000 rounds')
plt.show()

print ("Average payout after {} rounds is {}".format(num_rounds, sum(average_payouts)/(num_samples)))
