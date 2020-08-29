import time
import pickle
import gym
import numpy as np

# weights = np.zeros( (num_of_action, pow(num_of_param + 1, 2)) ) 

# 加载模型
with open('MountainCar_QLeaning-POLY.pickle', 'rb') as f:
    Q = pickle.load(f)
    print('model loaded')

env = gym.make('MountainCar-v0')
state_1_low, state_2_low  = env.observation_space.low
state_1_high, state_2_high = env.observation_space.high

class POLY_ValueFunction:
    def __init__(self, num_of_param, num_of_action):
        # W, 参数 parameter
        # 
        self.weights = np.copy( Q ) #  each function also has one more constant parameter (called bias in machine learning)
        # 每一个参数对应的 state 数量
        self.bases = [[], [], []]
        for action in [0, 1, 2]:
            for i in range(0, num_of_param+1):
                for j in range(0, num_of_param+1):
                    self.bases[action].append(lambda s_a, s_b, i=i, j=j: pow(s_a, i) * pow(s_b, j))


    def value(self, state_1, state_2, action):
        # map the state space into [0, 1]
        state_1 /= float(state_1_high)
        state_2 /= float(state_2_high)
 
        feature = np.asarray([func(state_1, state_2) for func in self.bases[action]])
        return np.dot(self.weights[action], feature)

    # update parameters
    def update(self, delta, state_1, state_2, action):
        # map the state space into [0, 1]
        state_1 /= float(state_1_high)
        state_2 /= float(state_2_high)

        # get derivative value
        derivative_value = np.asarray([func(state_1, state_2) for func in self.bases[action]] )
        self.weights[action] += delta * derivative_value



value_function = POLY_ValueFunction(6, 3)

s = env.reset()
score = 0
while True:
    env.render()
    time.sleep(0.01)

    a = 0 
    v_next = -100000
    for next_act in [0, 1, 2]:
        if v_next < value_function.value(s[0], s[1], next_act) :
            a = next_act
    s, reward, done, _ = env.step(a)
    score += reward
    if done:
        print('score:', score)
        break
env.close()