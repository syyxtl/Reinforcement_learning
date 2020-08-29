from collections import deque
import random
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers

class DQN(object):
    def __init__(self, observation_dim, action_n):
        self.INPUTDIM = observation_dim
        self.OUTPUTDIM = action_n

        self.step = 0
        self.update_freq = 200   # 模型更新频率 // target network 更新频率
        self.replay_size = 2000  # 训练集大小
        self.replay_memory = deque(maxlen=self.replay_size)  # replay memory
        self.eval_network = self.create_model()              # update network # 猫
        self.target_network = self.create_model()            # target network # 老鼠

    def create_model(self):
        '''
        网络的结构 depend on yourself
        映射 state -> action 的网络
        '''
        model = models.Sequential([
            layers.Dense(100, input_dim=self.INPUTDIM, activation='relu'),
            layers.Dense(self.OUTPUTDIM, activation="linear")
        ])
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001))
        return model

    def save_model(self, file_path='MountainCar-v0-dqn.h5'):
        print('model saved')
        self.eval_network.save(file_path)

    def remember(self, s, a, next_s, reward):
        """历史记录，position >= 0.4时给额外的reward，快速收敛"""
        if next_s[0] >= 0.4:
            reward += 1
        self.replay_memory.append((s, a, next_s, reward))

    def train(self, batch_size=64, lr=1, gamma=0.95):
        if len(self.replay_memory) < self.replay_size:
            return
        self.step += 1
        # 每 update_freq 步，将 model 的权重赋值给 target_network
        if self.step % self.update_freq == 0:
            self.target_network.set_weights(self.eval_network.get_weights())

        replay_batch = random.sample(self.replay_memory, batch_size)
        s_batch      = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])

        Q = self.eval_network.predict(s_batch)
        Q_next = self.target_network.predict(next_s_batch)

        # 使用公式更新训练集中的Q值
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + gamma * np.amax(Q_next[i]))
 
        # 传入网络进行训练
        self.eval_network.fit(s_batch, Q, verbose=0)

    def action_policy(self, s, epsilon=0.1):
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice([0, 1, 2])
        return np.argmax(self.eval_network.predict(np.array([s]))[0])

env = gym.make('MountainCar-v0')
agent = DQN(observation_dim=2, action_n=3)

def DQN_LEARN(env, agent, episodes_=1000):

    episodes = episodes_
    SCORES = []  # 记录所有分数

    for i in range(episodes):
        s = env.reset()
        score = 0
        while True:
            a = agent.action_policy(s)
            next_s, reward, done, _ = env.step(a)
            agent.remember(s, a, next_s, reward)
            agent.train()
            score += reward
            s = next_s
            if done:
                SCORES.append(score)
                print('episode:', i, 'score:', score, 'max:', max(SCORES))
                break
        # 最后10次的平均分大于 -160 时，停止并保存模型
        if np.mean(SCORES[-10:]) > -160:
            agent.save_model()
            break
    env.close()

    return SCORES


import matplotlib.pyplot as plt
SCORES = DQN_LEARN(env, agent)
plt.plot(SCORES, color='green')
plt.show()