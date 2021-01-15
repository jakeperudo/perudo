import config
import random
import sys
import time
from bet import DUDO
from AI import ComputerPlayer
from AI import HumanPlayer
from strings import correct_dudo
from strings import incorrect_dudo
from strings import INSUFFICIENT_BOTS
from strings import INSUFFICIENT_DICE
from strings import round_title
from strings import welcome_message

# "Burn all you love."
bot_names = ['Winston', 'Luke', 'Jeff', 'Jia', 'Ben']

class Perudo(object):

	def __init__(self, name, player_number, dice_number):
		self.round = 0
		self.total_dice = player_number * dice_number
		self.players = []
		self.players.append(
			HumanPlayer(
				name=name,
				dice_number=dice_number,
				game=self
			)
		)
		for i in range(0, player_number - 1):
			self.players.append(
				ComputerPlayer(
					name=self.get_random_name(),
					dice_number=dice_number,
					game=self
				)
			)

		random.shuffle(self.players)

		print (welcome_message(self.players))

		self.first_player = random.choice(self.players)

		odds = []
		rounds = []
		round_number=0

		while True:
			round_number += 1
			rounds.append(round_number)
			self.run_round()
			for j in range(0, player_number):
				if self.players[j].name == name:
					odds.append(self.players[j].score / round_number)


	def run_round(self):
		self.round += 1
		for player in self.players:
			player.roll_dice()

		print (round_title(round_number=self.round, is_palifico_round=self.is_palifico_round()))
		round_over = False
		current_bet = None
		current_player = self.first_player
		print ('{0} will go first...'.format(current_player.name))
		while not round_over:
			next_player = self.get_next_player(current_player)
			next_bet = current_player.make_bet(current_bet)
			bet_string = None


			if next_bet == DUDO:
				bet_string = 'Dudo!'
			else:
				bet_string = next_bet
			print ('{0}: {1}'.format(current_player.name, bet_string))
			if next_bet == DUDO:
				self.pause(0.5)
				self.run_dudo(current_player, current_bet)
				round_over = True
			else:
				current_bet = next_bet

			if len(self.players) > 1:
				current_player = next_player

			self.pause(0.5)

		self.pause(1)

	def run_dudo(self, player, bet):
		dice_count = self.count_dice(bet.value)
		if dice_count >= bet.quantity:
			print (incorrect_dudo(dice_count, bet.value))
			self.first_player = player
		else:
			print (correct_dudo(dice_count, bet.value))
			previous_player = self.get_previous_player(player)
			self.first_player = previous_player
		self.first_player.score += 1

	def count_dice(self, value):
		number = 0
		for player in self.players:
			number += player.count_dice(value)

		return number

	def is_palifico_round(self):
		if len(self.players) < 3:
			return False
		for player in self.players:
			if player.palifico_round == self.round:
				return True
		return False

	def get_random_name(self):
		random.shuffle(bot_names)
		return bot_names.pop()

	def get_next_player(self, player):
		return self.players[(self.players.index(player) + 1) % len(self.players)]

	def get_previous_player(self, player):
		return self.players[(self.players.index(player) - 1) % len(self.players)]

	def pause(self, duration):
		if config.play_slow:
			time.sleep(duration)

def get_argv(args, index, default):
	try:
		value = args[index]
	except IndexError:
		value = default
	return value

def main(args):
	name = get_argv(args, 1, 'Player')
	bot_number = int(get_argv(args, 2, 5))
	if bot_number < 1:
		print(INSUFFICIENT_BOTS)
		return
	dice_number = int(get_argv(args, 3, 6))
	if dice_number < 1:
		print(INSUFFICIENT_DICE)
		return

	Perudo(name, bot_number, dice_number)

if __name__ == '__main__':
	main(sys.argv)