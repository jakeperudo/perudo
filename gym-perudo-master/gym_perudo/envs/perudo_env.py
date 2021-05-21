import gym
from gym.spaces import Discrete
from gym.spaces import Tuple
import random
import numpy as np

from gym_perudo.envs.player import BotPlayer
from gym_perudo.envs.player import AIPlayer
import itertools

class PerudoEnv(gym.Env):

    def __init__(self):

        self.players = []
        self.dice_per_player = 0
        self.number_of_dice_sides = 0
        self.number_of_bots = 0
        self.bot_names=[]
        self.x = []
        self.dict = [(0,)]

        self.action_space = 0
        self.observation_space = 0

        self.render = False
        self.game_over = False
        self.round_over = False
        self.turn_over = False
        self.action = 0
        self.reward = 0

        self.round = 0
        self.previous_round = 0

        self.current_player = None
        self.next_player = None

        self.previous_previous_bet = 0
        self.previous_bet = 0
        self.current_bet = 0
        self.next_bet = 0

        self.remaining_dice = 0
        self.AI_dice = 0

    def step(self, action):

        self.game_over = False

        while self.game_over == False:

            if self.round > self.previous_round:
                if self.render == True:
                    print(' ')
                    print('Round {}'.format(self.round))
                for player in self.players:
                    player.roll_dice()
                for player in self.players:
                    if player.name == 'Mr AI':
                        self.AI_dice = self.dice_encoder(player)
                        if self.render == True:
                            diceList = []
                            for die in player.dice:
                                diceList.append(die.value)
                            print('Mr AI rolls : {}'.format(diceList))
                self.remaining_dice -= 1
                self.round_over = False
                self.turn_over = False
                self.previous_bet = 0
                self.current_player = self.players[random.randint(0,len(self.players)-1)]
                self.previous_round = self.round

            while self.round_over == False:

                self.turn_over = False

                while self.turn_over == False:

                    self.next_bet = self.current_player.make_bet(self.current_bet, action)

                    if self.current_player.name == 'Mr AI': #Check validity of AI bet
                        if self.invalidmove(self.current_player, self.current_bet, self.next_bet) == True:
                            self.reward = -10
                            if self.render == True:
                                bet_quantity, bet_value = self.bet_decoder(self.next_bet)
                                print('{} tried an incorrect play with {}x{}'.format(self.current_player.name, bet_quantity, bet_value))
                            self.ob_bet = self.current_bet
                            return self.get_obs(), self.reward, self.done
                        else:
                            self.turn_over = True
                    else:
                        self.turn_over = True  #Run through bets until valid bet is made


                    self.previous_previous_bet = self.previous_bet
                    self.previous_bet = self.current_bet # bet is valid
                    self.current_bet = self.next_bet # history of bets for dudo

                    if self.render == True:
                        if self.current_bet != 0:
                            bet_quantity, bet_value = self.bet_decoder(self.next_bet)
                            print('{} calls {} x {}'.format(self.current_player.name, bet_quantity, bet_value))

                if self.current_bet == 0: #Check if dudo and run
                    value = (self.previous_bet % self.number_of_dice_sides)
                    if value == 0:
                        value = self.number_of_dice_sides
                    quantity = ((self.previous_bet - value)//self.number_of_dice_sides) + 1
                    dice_count = self.count_dice(value)

                    if self.render == True:
                        print('{} calls dudo - there are actually {} x {}'.format(self.current_player.name, dice_count, value))

                    if dice_count >= quantity:
                        if self.render == True:
                            print('{} called dudo wrong and loses a die'.format(self.current_player.name))
                        if self.current_player.name == 'Mr AI':
                            self.reward = -2
                            self.ob_bet = self.previous_bet
                            self.remove_die(self.current_player)
                            if len(self.players) == 1:
                                self.done = True
                            return self.get_obs(), self.reward, self.done

                        if self.previous_player.name == 'Mr AI':
                            self.reward = 1
                            self.ob_bet = self.previous_previous_bet
                            self.remove_die(self.current_player)
                            if len(self.players) == 1:
                                self.done = True
                            return self.get_obs(), self.reward, self.done

                        else:
                            self.remove_die(self.current_player)


                    else:
                        if self.render == True:
                            print('{} called dudo right, {} loses a die'.format(self.current_player.name, self.previous_player.name))
                        if self.current_player.name == 'Mr AI':
                            self.reward = 1
                            self.ob_bet = self.previous_bet
                            self.remove_die(self.previous_player)
                            if len(self.players) == 1:
                                self.done = True
                            return self.get_obs(), self.reward, self.done

                        if self.previous_player.name == 'Mr AI':
                            self.reward = -2
                            self.ob_bet = self.previous_previous_bet
                            self.remove_die(self.previous_player)
                            if len(self.players) == 1:
                                self.done = True
                            return self.get_obs(), self.reward, self.done
                        else:
                            self.remove_die(self.previous_player)


                else: # If not dudo and is valid bet
                    self.previous_player = self.current_player
                    self.current_player = self.get_next_player(self.current_player)

            if len(self.players) == 1:
                self.game_over = True
                self.done = True


    def reset(self):

        self.done = False
        self.round = 1
        self.previous_round = 0
        self.first_bet = 0
        self.reward = 0
        self.ob_bet = 0
        self.ob_players = self.number_of_bots-1
        self.win = 1

        self.players = [AIPlayer(
                            name = 'Mr AI',
                            dice_number = self.dice_per_player,
                            game = self)]

        self.bot_names = ['Bot 4', 'Bot 3', 'Bot 2', 'Bot 1']

        for i in range(0,self.number_of_bots):
            self.players.append(
                BotPlayer(
                    name = self.get_random_name(),
                    dice_number = self.dice_per_player,
                    game = self))

        self.remaining_dice = len(self.players)*self.dice_per_player
        return self.get_obs()


    ##game functions##

    def get_next_player(self, player):
        return self.players[(self.players.index(player) + 1) % len(self.players)]

    def get_previous_player(self, player):
        return self.players[(self.players.index(player) - 1) % len(self.players)]

    def get_random_name(self):
        return self.bot_names.pop()

    def remove_die(self, player):
        self.previous_round = self.round
        self.round +=1
        self.round_over = True
        player.dice.pop()
        if len(player.dice) == 0:
            self.current_player = self.get_next_player(player)
            if self.render == True:
                print('{} is out!'. format(player.name))
            if player.name == 'Mr AI':
                self.done = True
                self.win = 0
                self.players.remove(player)
                return self.get_obs(), self.reward, self.done
            else:
                self.ob_players -=1
                self.players.remove(player)

    def count_dice(self, value):
        number = 0
        for player in self.players:
            number += player.count_dice(value)
        return number

    def get_obs(self):
        return self.ob_bet, self.remaining_dice, self.AI_dice

    def invalidmove(self, player, current_bet, next_bet):
        if player.name == 'Mr AI':
            if next_bet == 0:
                if current_bet == 0:
                    return True
                else:
                    return False
            else:
                if next_bet <= current_bet:
                    return True
                else:
                    return False

    def bet_decoder(self, bet):
        bet_value = (bet % self.number_of_dice_sides)
        if bet_value == 0:
            bet_value = self.number_of_dice_sides
        bet_quantity = ((bet - bet_value)//self.number_of_dice_sides) + 1
        return bet_quantity, bet_value

    def dice_encoder(self, player):
        current_dice = []
        for dice in player.dice:
            current_dice.append(dice.value)
        return self.dict.index(tuple(current_dice))
