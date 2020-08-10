import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb

# ENV --
WORLD_HEIGHT = 4
WORLD_WIDTH =12
NUM_STATE = WORLD_WIDTH * WORLD_HEIGHT

# ACTION --
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
NUM_ACTIONS = 4
ACTION = [UP, DOWN, RIGHT, LEFT]
ACTIONS = ['U', 'D', 'R', 'L']

# STATE --
START = (3,0)
END = (3,11)

def change_range(values, vmin=0, vmax=1):
	start_zero = values - np.min(values)
	return (start_zero / (np.max(start_zero) + 1e-7)) * (vmax - vmin) + vmin

class ENVIRONMENT:
	terrain_color = dict(normal=[127/360, 0, 96/100], objective=[26/360, 100/100, 100/100], cliff=[247/360, 92/100, 70/100], player=[344/360, 93/100, 100/100])

	def __init__(self):

		self.player = None
		self.num_steps = 0
		self.CreateEnv()
		self.DrawEnv()

	def CreateEnv(self, inital_grid = None):
		# create cliff walking grid world
		# just such as the following grid
		''' 0                    11
		0	x x x x x x x x x x x x
		1	x x x x x x x x x x x x
		2	x x x x x x x x x x x x
		3	S o o o o o o o o o o E
		'''
		self.grid = self.terrain_color['normal'] * np.ones((WORLD_HEIGHT, WORLD_WIDTH, 3)) 
		self.grid[-1, 1:11] = self.terrain_color['cliff']
		self.grid[-1,-1] = self.terrain_color['objective']

	def DrawEnv(self):
		self.fig, self.ax = plt.subplots(figsize=(WORLD_WIDTH, WORLD_HEIGHT))
		self.ax.grid(which='minor')       
		self.q_texts = [self.ax.text( i%WORLD_WIDTH, i//WORLD_WIDTH, '0', fontsize=11, verticalalignment='center', horizontalalignment='center') for i in range(12 * 4)]
		self.im = self.ax.imshow(hsv_to_rgb(self.grid), cmap='terrain', interpolation='nearest', vmin=0, vmax=1)
		self.ax.set_xticks(np.arange(WORLD_WIDTH))
		self.ax.set_xticks(np.arange(WORLD_WIDTH) - 0.5, minor=True)
		self.ax.set_yticks(np.arange(WORLD_HEIGHT))
		self.ax.set_yticks(np.arange(WORLD_HEIGHT) - 0.5, minor=True)  
		# plt.show()

	def step(self, action):
		# Possible actions
		if action == 0 and self.player[0] > 0:
			self.player = (self.player[0] - 1, self.player[1])
		if action == 1 and self.player[0] < 3:
			self.player = (self.player[0] + 1, self.player[1])
		if action == 2 and self.player[1] < 11:
			self.player = (self.player[0], self.player[1] + 1)
		if action == 3 and self.player[1] > 0:
			self.player = (self.player[0], self.player[1] - 1)

		self.num_steps = self.num_steps + 1
		# Rewards, game on
		# common situation: reward = -1 & game can carry on
		reward = -1
		done = False
		# if walk to the cliff, game over and loss, reward = -100
		# if walk to the destination, game over and win, reward = 0
		if self.player[0] == WORLD_HEIGHT-1 and self.player[1] > 0 and  self.player[1] < WORLD_WIDTH-1: 
			reward = -100
			done = True
		elif self.player[0] == END[0] and self.player[1] == END[1]:
			reward = 0
			done = True
		return self.player, reward, done

	def reset(self):
		self.player = [START[0], START[1]]
		self.num_steps = 0
		return self.player

	def RenderEnv(self, q_values, action=None, max_q=False, colorize_q=False):
		assert self.player is not None, 'You first need to call .reset()'

		if colorize_q:       
			grid = self.terrain_color['normal'] * np.ones((4, 12, 3))
			values = change_range(np.max(q_values, -1)).reshape(4, 12)
			grid[:, :, 1] = values
			grid[-1, 1:11] = self.terrain_color['cliff']
			grid[-1,-1] = self.terrain_color['objective']
		else:
			grid = self.grid.copy()

		# render the player grid
		grid[self.player] = self.terrain_color['player']       
		self.im.set_data(hsv_to_rgb(grid))

		if q_values is not None:
			xs = np.repeat(np.arange(12), 4)
			ys = np.tile(np.arange(4), 12)  

			for i, text in enumerate(self.q_texts):
				txt = ""
				for aaction in range(len(ACTIONS)):
					txt += str(ACTIONS[aaction]) + ":" + str( round(q_values[ i//WORLD_WIDTH, i%WORLD_WIDTH, aaction], 2) ) + '\n'
				text.set_text(txt)
		# show the action

		if action is not None:
			self.ax.set_title(action, color='r', weight='bold', fontsize=32)

		plt.pause(0.1)



def egreedy_policy( q_values, state, epsilon=0.1):
	if np.random.binomial(1, epsilon) == 1:
		return np.random.choice(ACTION)
	else:
		values_ = q_values[state[0], state[1], :]
		return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

def sarsa(env, episodes=500, render=True, epsilon=0.1, learning_rate=0.5, gamma=0.9):
	q_values_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, NUM_ACTIONS))
	ep_rewards = []
	# Qlearning begin...
	for _ in range(0,episodes):
		state = env.reset()
		done = False
		reward_sum = 0
		action = egreedy_policy(q_values_sarsa, state, epsilon)

		while done == False:
			next_state, reward, done = env.step(action)
			next_action = egreedy_policy(q_values_sarsa, next_state, epsilon)
			# 普通sarsa
			q_values_sarsa[state[0], state[1], action] +=  learning_rate * (reward + gamma * q_values_sarsa[next_state[0], next_state[1], next_action] - q_values_sarsa[state[0], state[1], action])
			# 期望 sarsa
			# q_values_sarsa[state[0], state[1], action] +=  learning_rate * (reward + gamma * q_values_sarsa[next_state[0], next_state[1], next_action] - q_values_sarsa[state[0], state[1], action])
			state = next_state
			action = next_action
			# for comparsion, record all the rewards, this is not necessary for QLearning algorithm
			reward_sum += reward

			if render:
				env.RenderEnv(q_values_sarsa, action=ACTIONS[action], colorize_q=True)

		ep_rewards.append(reward_sum)
	# Qlearning end...
	return ep_rewards, q_values_sarsa

def play(q_values):
	# simulate the environent using the learned Q values
	env = ENVIRONMENT()
	state = env.reset()
	done = False

	while not done:    
		# Select action
		action = egreedy_policy(q_values, state, 0.0)
		# Do the action
		state_, R, done = env.step(action)  
		# Update state and action        
		state = state_  
		env.RenderEnv(q_values=q_values, action=ACTIONS[action], colorize_q=True)


env = ENVIRONMENT()
sarsa_rewards, q_values_sarsa = sarsa(env, episodes=500, render=False, epsilon=0.1, learning_rate=1, gamma=0.9)
play(q_values_sarsa)