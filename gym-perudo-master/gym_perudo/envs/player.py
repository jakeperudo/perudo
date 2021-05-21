import random
from math import floor
from math import ceil
from random import randrange
import numpy as np

class BetException(Exception):
	pass


class InvalidBetException(BetException):
	#Raised when a bet does not have either a higher quantity or a higher value
	pass


class Bet(object):
	def __init__(self, bet):
		self.current_bet = bet
	def __repr__(self):
		return 'bet'

def create_bet(proposed_bet, last_bet, player, game):
	if last_bet:
		if proposed_bet <= last_bet:
			raise InvalidBetException()
		return proposed_bet
	else:
		return proposed_bet


class Die(object):
	def __init__(self, game):
		self.roll(game)
	def roll(self, game):
		self.value = randrange(1,game.number_of_dice_sides+1)


class Player(object):

	def __init__(self, name, dice_number, game):
		self.name = name
		self.game = game
		self.dice = []
		for i in range(0, dice_number):
			self.dice.append(Die(game))

	def make_bet(self, current_bet, action):
		pass

	def roll_dice(self):
		for die in self.dice:
			die.roll(self.game)
		self.dice = sorted(self.dice, key=lambda die: die.value)

	def count_dice(self, value):
		number = 0
		for die in self.dice:
			if die.value == value:
				number += 1
		return number

class BotPlayer(Player):

	def make_bet(self, current_bet, action):

		total_dice_estimate = self.game.remaining_dice

		if current_bet == 0:
			value = randrange(1, self.game.number_of_dice_sides + 1) #random.choice(self.dice).value
			quantity_limit = (total_dice_estimate - len(self.dice))//self.game.number_of_dice_sides
			quantity = random.randrange(1, 3)
			bet = self.game.number_of_dice_sides*(quantity-1) + value
			bet = create_bet(bet, current_bet, self, self.game)
			return bet

		else:
			value = (current_bet % self.game.number_of_dice_sides)
			if value == 0:
				value = self.game.number_of_dice_sides
			quantity = ((current_bet - value)//self.game.number_of_dice_sides) + 1
			limit = ceil(total_dice_estimate/self.game.number_of_dice_sides) + random.randrange(0, ceil(total_dice_estimate/4.0))

			if quantity >= limit:
				bet = 0
				return bet
			else:
				bet = None
				while bet is None:
					value = randrange(1,self.game.number_of_dice_sides+1)  #random.choice(self.dice).value
					quantity = quantity + random.randrange(0, 2)
					bet = self.game.number_of_dice_sides*(quantity-1) + value
					try:
						bet = create_bet(bet, current_bet, self, self.game)
					except BetException:
						bet = None

				value = (bet % self.game.number_of_dice_sides)
				if value == 0:
					value = self.game.number_of_dice_sides
				quantity = ((bet - value)//self.game.number_of_dice_sides) + 1
			return bet


class AIPlayer(Player):
	def make_bet(self, current_bet, action):
		return action
