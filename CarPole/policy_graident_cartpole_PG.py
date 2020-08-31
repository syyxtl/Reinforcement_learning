import matplotlib.pyplot as plt
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers



class PG(object):
    def __init__(self, observation_dim, action_n):
        self.INPUTDIM = observation_dim
        self.OUTPUTDIM = action_n

        self.trajectory = [] # 轨迹 s-a-r
        self.PG_NET = self.create_model()

    def create_model(self):
        model = models.Sequential([
            layers.Dense(100, input_dim=self.INPUTDIM, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(self.OUTPUTDIM, activation="softmax")
        ])
        model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.001))
        return model

    def save_model(self, file_path='CartPole-v0-PG.h5'):
        print('model saved')
        self.PG_NET.save(file_path)

    def remember(self, s, a, r):
        self.trajectory.append((s, a, r))

    def discount_rewards(self, rewards, gamma=0.95):
        """计算衰减reward的累加期望，并中心化和标准化处理"""
        prior = 0
        out = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            prior = prior * gamma + rewards[i]
            out[i] = prior
        return out / np.std(out - np.mean(out))

    def train(self, gamma=0.95):
        # X=state
        # Y=action prob
        s_batch = np.array([traj[0] for traj in self.trajectory]) # 取 state
        a_batch = np.array([traj[1] for traj in self.trajectory])
        r_batch = np.array([traj[2] for traj in self.trajectory])

        # 计算带折扣的reward之和=G
        G = self.discount_rewards(r_batch)

        # discount = np.power(gamma, list(range(0,len(r_batch))))
        # discount_reward = discount * r_batch
        # G = discount_reward[::-1].cumsum()
        # G = G / np.std(G - np.mean(G))

        # Y: softnax ---> a_batch_one_hot
        a_batch_one_hot = np.eye(self.OUTPUTDIM)[a_batch]
        prob_batch = self.PG_NET.predict(s_batch) * a_batch_one_hot
        # Y = a_batch_one_hot * G[:, np.newaxis]
        # self.PG_NET.fit(s_batch, Y, verbose=0)
        self.PG_NET.fit(s_batch, prob_batch, sample_weight=G, verbose=0)
        self.trajectory = []

    def action_policy(self, state):  #根据策略函数选择行为
        prob = self.PG_NET.predict(np.array([state]))[0]
        return np.random.choice(len(prob), p=prob)
        
env = gym.make('CartPole-v0')
agent = PG(observation_dim=4, action_n=2)

def PG_LEARN(env, agent, episodes_=1000):

    episodes = episodes_
    SCORES = []  # 记录所有分数

    for i in range(episodes):
        s = env.reset()
        score = 0

        while True:
            a = agent.action_policy(s)
            next_s, r, done, _ = env.step(a)
            agent.remember(s, a, r)
            score += r
            s = next_s

            if done:
                agent.train()
                SCORES.append(score)
                print('episode:', i, 'score:', score, 'max:', max(SCORES))
                break
        # 最后10次的平均分大于 195 时，停止并保存模型
        if np.mean(SCORES[-10:]) > 195:
            agent.save_model()
            break
    
    env.close()
    return SCORES


SCORES = PG_LEARN(env, agent)