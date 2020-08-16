import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# value prediction for MC and TD(0)
# 1000 state随机步游，共有1000个位置，中心位置500为起点，最左侧和最右侧为终点，
# 每次行动会向左侧或者右侧的100个位置中随机选择一个位置，
# 达到边缘的时候，到达不存在的位置时，其概率等于边界终止的概率
# 达到最右侧 reward=1，到达最左侧rewards=-1， 其余都为0

# ENV --
LENGTH = 1000

# ACTION --
LEFT = 0
RIGHT = 1
ACTIONS = [LEFT, RIGHT]
ACTION = ["LEFT", "RIGHT"]

# SATAE --
START = 500
END = 1001

RANGES = 100

class ENVIRONMENT:

    def __init__(self):
        self.player = None # player is corresponding to STATE
        self.num_steps = 0
        self.CreateEnv()
        self.DrawEnv()
    
    def CreateEnv(self, inital_grid = None):
        self.grid = ["o"] * 1000

    def DrawEnv(self):
        # print(self.grid)
        pass

    def step(self, action):
        # random choice actiion
        step = np.random.randint(1, RANGES + 1)
        if action == 0:
            step = step * -1
        else:
            step = step * 1

        self.player = self.player + step
        self.player = max(min(self.player, LENGTH+1),0)
        
        self.num_steps = self.num_steps + 1
        # Rewards, game on
        reward = 0
        done = False

        if self.player == 0: 
            reward = -1
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
            self.grid = ["o"] * (LENGTH + 2)
            self.grid[ self.player ] = "*"
            print(self.grid)

def action_policy(epsilon=0.5):
    if np.random.binomial(1, epsilon) == LEFT:
        return LEFT
    else:
        return RIGHT

def compute_true_value_DP(episodes=1000):
    true_value = np.zeros(1002)
    
    # while True:
    for i in range(episodes):
        old_value = np.copy(true_value)
        for state in range(LENGTH+1): # 对每一个state 循环，计算其 value function
            # 求和函数
            true_value[state] = 0
            for action in ACTIONS:
                for step in range(RANGES):
                    if action == 0:
                        step = step * -1
                    else:
                        step = step * 1
                    next_state= state + step
                    next_state = max(min(next_state, LENGTH+1),0)
                    reward = 0
                    if next_state == 0:
                        reward = -1
                    elif next_state == LENGTH+1:
                        reward = 1
                    gamma = 1
                    true_value[state] += 1.0 / (2 * RANGES) * ( reward + gamma * true_value[next_state] )

        error = np.sum( np.abs(true_value-old_value) )
        if i % 50 == 0:
            print(error)
        if error < 1e-2:
            break

    true_value[0] = true_value[-1] = 0
    return true_value


# 测试不同episodes下价值函数的值，用于比较与真值之间的差别
def test_true_value():
    plt.figure(1)
    values = compute_true_value_DP()
    plt.plot(values, label='DP true values')
    
    plt.xlabel('state')
    plt.ylabel('estimated value')

    plt.legend()
    plt.show()


 # 多项式基 作为 拟合特征
class TilingsValueFunction:
    # @num_of_tilings: # of tilings
    # @tileWidth: each tiling has several tiles, this parameter specifies the width of each tile
    # @tilingOffset: specifies how tilings are put together
    def __init__(self, numOfTilings, tileWidth, tilingOffset):
        self.numOfTilings = numOfTilings
        self.tileWidth = tileWidth
        self.tilingOffset = tilingOffset

        # To make sure that each sate is covered by same number of tiles,
        # we need one more tile for each tiling
        self.tilingSize = LENGTH // tileWidth + 1

        # weight for each tile
        self.params = np.zeros((self.numOfTilings, self.tilingSize))

        # For performance, only track the starting position for each tiling
        # As we have one more tile for each tiling, the starting position will be negative
        self.tilings = np.arange(-tileWidth + 1, 0, tilingOffset)

    # get the value of @state
    def value(self, state):
        stateValue = 0.0
        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            stateValue += self.params[tilingIndex, tileIndex]
        return stateValue

    # update parameters
    # @delta: step size * (target - old estimation)
    # @state: state of current sample
    def update(self, delta, state):

        # each state is covered by same number of tilings
        # so the delta should be divided equally into each tiling (tile)
        delta /= self.numOfTilings

        # go through all the tilings
        for tilingIndex in range(0, len(self.tilings)):
            # find the active tile in current tiling
            tileIndex = (state - self.tilings[tilingIndex]) // self.tileWidth
            self.params[tilingIndex, tileIndex] += delta

## gradient monte carlo algorithm
def gradient_monte_carlo(env, episodes_, epsilon=0.5, learning_rate=2e-5, gamma=1):
    value_function = TilingsValueFunction(50, 200, 4) # 10个参数的 function

    episode = episodes_

    for _ in range(episode):
        state = env.reset()
        trajectory = [START]
        
        done = False
        reward = 0.0
        REWARDS = [0]
        # We assume gamma = 1, so return is just the same as the latest reward
        while done == False:
            action = action_policy(epsilon)
            next_state, reward, done = env.step(action)
            trajectory.append(next_state)
            state = next_state
            REWARDS.append(reward)

        G = 0
        W = 1
        # Gradient update for each state in this trajectory
        # print(trajectory[:-1], reward)
        for state_ in trajectory[:-1]:
            delta = learning_rate * (reward - value_function.value(state_))
            value_function.update(delta, state_)

    return value_function
# true_value = compute_true_value_DP()

def PIC():
    env = ENVIRONMENT()

    values_eval = gradient_monte_carlo(env, episodes_=5000)
    state_values = [values_eval.value(i) for i in np.arange(1, LENGTH + 1)]

    plt.figure(1)
    plt.plot(state_values, label='Approximate TD value')
    # plt.plot(true_value, label='true value')
    plt.xlabel('State')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

PIC()