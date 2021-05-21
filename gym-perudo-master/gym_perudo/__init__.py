from gym.envs.registration import register

register(
	id='perudo_game-v0',
	entry_point='gym_perudo.envs:PerudoEnv')
