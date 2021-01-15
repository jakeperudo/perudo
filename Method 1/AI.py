import random
import numpy as np
from scipy.special import comb
from bet import DUDO
from bet import create_bet
from bet_exceptions import BetException
from bet_exceptions import InvalidDieValueException
from bet_exceptions import NonPalificoChangeException
from bet_exceptions import InvalidNonWildcardQuantityException
from bet_exceptions import InvalidWildcardQuantityException
from bet_exceptions import InvalidBetException
from die import Die
from math import ceil
from strings import BAD_BET_ERROR
from strings import INVALID_DIE_VALUE_ERROR
from strings import NON_PALIFICO_CHANGE_ERROR
from strings import INVALID_NON_WILDCARD_QUANTITY
from strings import INVALID_WILDCARD_QUANTITY
from strings import INVALID_BET_EXCEPTION

Lambda = 0.9
alpha = 0.1
epsilon = 1
epsilon_decay = 0.99

class AI(object):

    def __init__(self, name, dice_number, game):
        self.name = name
        self.game = game
        self.score = 0 # +1 when win a round
        self.palifico_round = -1
        self.dice = []
        for i in range(0, dice_number):
            self.dice.append(Die())
        self.total_dice = self.game.total_dice
        self.table = np.zeros((self.game.total_dice, 6, self.game.total_dice + 1, 7))
        # build a 4-D Q table with quantity, value, quantity action(dudo, +0, +1...) and value action(value -> value/dudo)
        # position [x,y,0,0] are all dudo



    def roll_dice(self):
        for die in self.dice:
            die.roll()
        # Sort dice into value order e.g. 4 2 5 -> 2 4 5
        self.dice = sorted(self.dice, key=lambda die: die.value)

    def count_dice(self, value):
        # same as perudo master
        number = 0
        for die in self.dice:
            if die.value == value or (not self.game.is_palifico_round() and die.value == 1):
                number += 1
        return number


    def prob(self, dice_amount, num_amount,number):
        # the probability of the bet which is Binomial distribution
        P = comb(dice_amount, num_amount) * (1 - number / 6) * ((1 / 3) ** num_amount) * ((2 / 3) ** (dice_amount - num_amount))
        for k in range(num_amount + 1, dice_amount):
            P += comb(dice_amount, k) * ((1 / 3) ** k) * ((2 / 3) ** (dice_amount - k))
        if number > 6 or num_amount > self.total_dice:
            # avoid incorrect bet
            P = 0
        return P

class ComputerPlayer(AI):

    def choose_action(self, q_table, current_bet):
        # action choice is not just dependent on Q table value but on expect value (Q table value x probability)
        P_current=self.prob(self.total_dice, current_bet.quantity-self.count_dice(current_bet.value), current_bet.value)
        # probability of current bet
        Max = (1 - P_current) * q_table[current_bet.quantity-1, current_bet.value-1, 0, 0]
        locationx = 0
        locationy = 0
        value = random.choice(self.dice).value

        global epsilon
        epsilon *= epsilon_decay
        
        if P_current < 0.1:
            return DUDO

        else:
            if np.random.uniform() < epsilon:
                # choose randomly at start when Q table is not complete
                return random.randrange(1, 3), value

            else:
                # go through the Q table to find the max expect value with its location
                for k in range(1,self.total_dice+2-current_bet.quantity):
                    pre_q = current_bet.quantity + k

                    if k == 1:
                        for l in range(current_bet.value + 1, 7):
                            pre_v = l
                            P = self.prob(self.total_dice, pre_q - self.count_dice(pre_v), pre_v)
                            if q_table[current_bet.quantity - 1, current_bet.value - 1, k, l] * P > Max:
                                Max = q_table[current_bet.quantity - 1, current_bet.value - 1, k, l] * P
                                locationx, locationy = k, l

                    else:
                        for l in range(1, 7):
                            pre_v = l
                            P = self.prob(self.total_dice, pre_q - self.count_dice(pre_v), pre_v)
                            if q_table[current_bet.quantity - 1, current_bet.value - 1, k, l] * P > Max:
                                Max = q_table[current_bet.quantity - 1, current_bet.value - 1, k, l] * P
                                locationx, locationy = k, l

                if Max == (1 - P_current) * q_table[current_bet.quantity-1, current_bet.value-1, 0, 0]:
                    if Max == 0:
                        # if Q table still not complete, choose randomly
                        return random.randrange(1, 3), value
                    else:
                        return DUDO
                else:
                    return locationx, locationy

    def reward(self, bet):
        # check whether the bet is valid
        dice_count = self.game.count_dice(bet.value)
        if dice_count >= bet.quantity:
            re = 0
        else:
            re = 1
        return re


    def Qtable(self,current_bet,bet):
        # update Q table
        if self.choose_action(self.table, current_bet) != DUDO:
            Action = self.choose_action(self.table, current_bet)
            q_predict = self.table[current_bet.quantity-1, current_bet.value-1, Action[0], Action[1]]

            # find the max Q table value
            Max = 0
            for k in range(0,self.total_dice+1):
                for l in range(0,7):
                    pre_q = current_bet.quantity + Action[0]
                    pre_v = Action[1]
                    if self.table[pre_q - 2, pre_v - 1, k, l] > Max:
                        Max = self.table[pre_q -1, pre_v -1, k, l]

            q_target = (1 - self.reward(bet)) + Lambda * Max
            # system will check if the bet satisfy dudo, if satisfy, reward = 0, otherwise reward = 1
            self.table[current_bet.quantity-1, current_bet.value-1, Action[0], Action[1]] += alpha * (q_target - q_predict)

        else:
            q_target = self.reward(current_bet) * 10 # if correct dudo, reward = 10
            q_predict = self.table[current_bet.quantity-1, current_bet.value-1, 0, 0]
            self.table[current_bet.quantity-1, current_bet.value-1, 0, 0] += alpha * (q_target - q_predict)


    def make_bet(self, current_bet):

        if current_bet is None:
            value = random.choice(self.dice).value
            quantity_limit = (self.total_dice - len(self.dice)) / 6

            if value > 1:
                quantity_limit *= 2

            quantity = self.count_dice(value) + random.randrange(0, ceil(quantity_limit + 1))
            bet = create_bet(quantity, value, current_bet, self, self.game)

        else:
            if self.choose_action(self.table, current_bet) == DUDO:
                return DUDO

            else:
                Action = self.choose_action(self.table, current_bet)
                act_q = current_bet.quantity + Action[0]  # quantity + action = next bet's quantity
                act_v = Action[1]
                bet = create_bet(act_q, act_v, current_bet, self, self.game)

            self.Qtable(current_bet, bet)
        return bet


class HumanPlayer(AI):

    def make_bet(self, current_bet):
        string = 'Your turn. Your dice:'
        for die in self.dice:
            string += ' {0}'.format(die.value)
        print (string)
        bet = None
        while bet is None:
            bet_input = input('> ')
            if bet_input.lower() == 'dudo':
                return DUDO
            if 'x' not in bet_input:
                print (BAD_BET_ERROR)
                continue
            bet_fields = bet_input.split('x')
            if len(bet_fields) < 2:
                print (BAD_BET_ERROR)
                continue

            try:
                quantity = int(bet_fields[0])
                value = int(bet_fields[1])

                try:
                    bet = create_bet(quantity, value, current_bet, self, self.game)
                except InvalidDieValueException:
                    bet = None
                    print (INVALID_DIE_VALUE_ERROR)
                except NonPalificoChangeException:
                    bet = None
                    print (NON_PALIFICO_CHANGE_ERROR)
                except InvalidNonWildcardQuantityException:
                    bet = None
                    print (INVALID_NON_WILDCARD_QUANTITY)
                except InvalidWildcardQuantityException:
                    bet = None
                    print (INVALID_WILDCARD_QUANTITY)
                except InvalidBetException:
                    bet = None
                    print (INVALID_BET_EXCEPTION)
            except ValueError:
                print (BAD_BET_ERROR)

        return bet

class RandomPlayer(AI):
    def make_bet(self, current_bet):
        if current_bet is None:
            value = random.choice(self.dice).value
            quantity_limit = (self.total_dice - len(self.dice)) / 6

            if value > 1:
                quantity_limit *= 2

            quantity = self.count_dice(value) + random.randrange(0, ceil(quantity_limit + 1))
            bet = create_bet(quantity, value, current_bet, self, self.game)

        else:
            # Estimate the probability of current bet
            P_current = self.prob(self.total_dice, current_bet.quantity - self.count_dice(current_bet.value),current_bet.value)
            # Estimate the number of dice in the game with the bet's value
            if current_bet.value == 1 or self.game.is_palifico_round():
                # There should be twice as many of any value than 1
                limit = ceil(self.total_dice / 6.0) + random.randrange(0, ceil(self.total_dice / 4.0))
            else:
                limit = ceil(self.total_dice / 6.0) * 2 + random.randrange(0, ceil(self.total_dice / 4.0))
            if current_bet.quantity >= limit or P_current < 0.1:
                return DUDO
            else:
                bet = None
                while bet is None:
                    if self.game.is_palifico_round() and self.palifico_round == -1:
                        # If it is a Palifico round and the player has not already been palifico,
                        # the value cannot be changed.
                        value = current_bet.value
                        quantity = current_bet.quantity + 1
                    else:
                        value = random.choice(self.dice).value
                        if value == 1:
                            if current_bet.value > 1:
                                quantity = int(ceil(current_bet.quantity / 2.0))
                            else:
                                quantity = current_bet.quantity + 1
                        else:
                            if current_bet.value == 1:
                                quantity = current_bet.quantity * 2 + 1
                            else:
                                quantity = current_bet.quantity + 1

                    try:
                        bet = create_bet(quantity, value, current_bet, self, self.game)
                    except BetException:
                        bet = None

        return bet

class TotalRandom(AI):
    def make_bet(self, current_bet):
        if current_bet is None:
            value = random.choice(self.dice).value
            quantity_limit = (self.total_dice - len(self.dice)) / 6

            if value > 1:
                quantity_limit *= 2

            quantity = self.count_dice(value) + random.randrange(0, ceil(quantity_limit + 1))
            bet = create_bet(quantity, value, current_bet, self, self.game)

        else:
            if np.random.uniform() < 0.2:
                return DUDO
            else:
                value = random.choice(self.dice).value
                quantity = current_bet.quantity + 1
                try:
                    bet = create_bet(quantity, value, current_bet, self, self.game)
                except BetException:
                    bet = None

        return bet