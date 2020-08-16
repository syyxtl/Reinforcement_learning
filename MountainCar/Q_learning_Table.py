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

def transform_state(env, state):
    """将 position, velocity 通过线性转换映射到 [0, 40] 范围内"""
    pos, v = state
    pos_low, v_low = env.observation_space.low
    pos_high, v_high = env.observation_space.high

    a = 40 * (pos - pos_low) / (pos_high - pos_low)
    b = 40 * (v - v_low) / (v_high - v_low)

    return int(a), int(b)

def action_policy(ep, episodes, q_values, state):
    if np.random.random() > ep * 3 / episodes:
        return np.random.choice([0, 1, 2])
    else:
        return np.argmax(q_values[state[0]][state[1]])
        # values_ = q_values[state[0], state[1], :]
        # return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])




def Qlearning_TABLE(env, episodes_=10000,learning_rate=0.7, gamma=0.95):
    q_values = np.zeros((41, 41, 3))
    episodes = episodes_

    SCORES = []

    for ep in range(episodes):
        state = env.reset()
        state = transform_state(env, state)
        score = 0
        while True:
            action = action_policy(ep, episodes, q_values, state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(env, next_state)
            q_values[state[0]][state[1]][action] += learning_rate * ( reward + gamma * max(q_values[next_state[0], next_state[1], :]) - q_values[state[0]][state[1]][action] ) 
            state = next_state
            score += reward
            if done:
                SCORES.append(score)
                print('episode:', ep, 'score:', score, 'max:', max(SCORES))
                break

    return q_values, SCORES

env = gym.make('MountainCar-v0')
q_values, _ = Qlearning_TABLE(env)
env.close()


# 保存模型
with open('MountainCar_QLeaning-TABLE.pickle', 'wb') as f:
    pickle.dump(q_values, f)
    print('model saved')
