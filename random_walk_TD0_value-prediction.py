import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# value prediction for MC and TD(0)
# 随机步游，共有7个位置，ABCDEFG，D为起点，G为终点，到达A或者G游戏结束
# 达到G reward=1，其余都为0

# ENV --
LENGTH = 7

# ACTION --
LEFT = 0
RIGHT = 1
ACTIONS = [LEFT, RIGHT]
ACTION = ["LEFT", "RIGHT"]

# SATAE --
START = 3
END = 6

# VALUES --
VALUES = np.zeros(LENGTH)
VALUES[1:END] = 0.5
VALUES[END] = 0
# SET TRUE VALUES --
TRUE_VALUE = np.zeros(7)
TRUE_VALUE[1:END] = np.arange(1, 6) / 6.0
TRUE_VALUE[END] = 0


class ENVIRONMENT:

	def __init__(self):

		self.player = None # player is corresponding to STATE
		self.num_steps = 0
		self.CreateEnv()
		self.DrawEnv()
	
	def CreateEnv(self, inital_grid = None):
		''' environment is like the following:
			o o o S o o E
		'''
		self.grid = ["o"] * 7

	def DrawEnv(self):
		print(self.grid)

	def step(self, action):
		# Possible actions
		if action == 0 and self.player > 0:
			self.player = self.player - 1
		if action == 1 and self.player < END:
			self.player = self.player + 1
		
		self.num_steps = self.num_steps + 1
		# Rewards, game on
		# common situation: reward = -1 & game can carry on
		reward = 0
		done = False
		# if walk to the cliff, game over and loss, reward = -100
		# if walk to the destination, game over and win, reward = 0
		if self.player == 0: 
			reward = 0
			done = True
		elif self.player == END:
			reward = 1
			done = True
		return self.player, reward, done

	def reset(self):
		self.player = START
		self.num_steps = 0
		return self.player

	def RenderEnv(self, action=None, render=False):
		if render:
			self.grid[ self.player ] = "*"
			print(self.grid)

def action_policy(epsilon=0.5):
	if np.random.binomial(1, epsilon) == LEFT:
		return LEFT
	else:
		return RIGHT


import copy
def TD0(env, episodes_, epsilon=0.5, learning_rate=0.1, gamma=1):
	values = copy.deepcopy(VALUES)
	episode = episodes_

	for _ in range(episode):
		state = env.reset()
		done = False
		while done == False:
			action = action_policy()
			next_state, reward, done = env.step(action)
			values[state] += learning_rate * (reward + gamma * values[next_state] - values[state])
			state = next_state

	return values


# 测试不同episodes下价值函数的值，用于比较与真值之间的差别
def test_true_value():
	episodes = [0, 1, 10, 100]
	plt.figure(1)
	for _ in episodes:
		env = ENVIRONMENT()
		values = TD0(env, episodes_=int(_), epsilon=0.5, learning_rate=0.1, gamma=0.9)
		plt.plot(values, label=str(_) + ' episodes')
	
	plt.plot(TRUE_VALUE, label='true values')
	plt.xlabel('state')
	plt.ylabel('estimated value')

	plt.xlim(1, 5)
	plt.legend()
	plt.show()

test_true_value()