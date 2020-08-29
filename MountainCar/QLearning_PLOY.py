import gym
import pickle
import numpy as np
from collections import defaultdict

'''
环境再 gym中已经编好，不再另外编写环境 https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
为了理解，说明环境中的一些必要知识
actions:
action = {0，1，2}
0 : 加速向左
1 : 不加速
2 : 加速向右

observation:
每一个状态拥有两个观测值，即每一个状态有两个数字表示（一般来说，在强化学习中 状态由若干个数字表示，这些数字拥有某些物理意义）：
0 : position 在 ( -1.2, 0.6)  之间
1 : velocity 在 (-0.07, 0.07) 之间

rewards:
到达终点 reward = 0
otherwise reward = -1

states:二元组 (position, velocity)
初始值 postion = [-0.6, -0.4]
       velocity = 0
终止 postion >= 0.5
episode > 200

具体的速度和位置关系 与力的关系 需要根据 物理知识进行推导
'''
env = gym.make('MountainCar-v0')
state_1_low, state_2_low  = env.observation_space.low
state_1_high, state_2_high = env.observation_space.high

 # 多项式基 作为 拟合特征
class POLY_ValueFunction:
    def __init__(self, num_of_param, num_of_action):
        # W, 参数 parameter
        # 
        self.weights = np.zeros( (num_of_action, pow(num_of_param + 1, 2)) ) #  each function also has one more constant parameter (called bias in machine learning)
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


def action_policy(state, value_function):
    if np.random.binomial(1, 0.3) == 1:
        return np.random.choice([0, 1, 2])
    values = []
    for action in [0, 1, 2]:
        values.append(value_function.value(state[0], state[1], action))
    return np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

def SARSA_POLY(env, episodes_=3000, learning_rate=2e-4, gamma=1):
    value_function = POLY_ValueFunction(6, 3) # 多项式 最多次数为 5, action num = 3
    
    episode = episodes_
    SCORES = []

    for ep in range(episode):
        state = env.reset()
        action = action_policy(state, value_function)
        score = 0
        while True:
            next_state, reward, done, _ = env.step(action)
            score += reward
            if done:
                SCORES.append(score)
                v_this = value_function.value(state[0], state[1], action)
                delta = learning_rate * (reward - v_this)
                value_function.update(delta, state[0], state[1], action)
                print('episode:', ep, 'score:', score, 'max:', max(SCORES))
                break
            next_action = action_policy(next_state, value_function) 
            v_this = value_function.value(state[0], state[1], action)
            v_next = value_function.value(next_state[0], next_state[1], next_action)
            delta = learning_rate * (reward + gamma* v_next - v_this)
            value_function.update(delta, state[0], state[1], action)
            state = next_state
            action = next_action

    return value_function, SCORES

def Qlearning_POLY(env, episodes_=10,learning_rate=1e-4, gamma=0.95):
    print(learning_rate)
    value_function = POLY_ValueFunction(5, 3) # 多项式 最多次数为 5, action num = 3

    episodes = episodes_

    SCORES = []

    for ep in range(episodes):
        state = env.reset()
        score = 0
        while True:
            action = action_policy(state, value_function)
            next_state, reward, done, _ = env.step(action)

            v_this = value_function.value(state[0], state[1], action)
            v_next = -10000
            for next_act in [0, 1, 2]:
                v_next = max(value_function.value(next_state[0], next_state[1], next_act), v_next)
            delta = learning_rate * (reward + gamma* v_next - v_this)
            value_function.update(delta, state[0], state[1], action)
            # q_values[state[0]][state[1]][action] += learning_rate * ( reward + gamma * max(q_values[next_state[0], next_state[1], :]) - q_values[state[0]][state[1]][action] ) 
            state = next_state
            score += reward
            if done:
                SCORES.append(score)
                # print(value_function.weights)
                print('episode:', ep, 'score:', score, 'max:', max(SCORES))
                break

    return value_function, SCORES

value_function, _ = SARSA_POLY(env)
env.close()


# 保存模型
with open('MountainCar_QLeaning-POLY.pickle', 'wb') as f:
    pickle.dump(value_function.weights, f)
    print('model saved')
