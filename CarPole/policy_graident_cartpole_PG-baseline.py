import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
from tensorflow.keras import models, layers, optimizers



class PG(object):
    def __init__(self, observation_dim, action_n):
        self.INPUTDIM = observation_dim
        self.OUTPUTDIM = action_n

        self.trajectory = [] # 轨迹 (s-a-r)
        # 参数： 输出维度， 最后一层激活函数，损失函数
        self.PG_NET = self.create_model(self.OUTPUTDIM, "softmax", "categorical_crossentropy")
        self.BASELINE = self.create_model(1, None, "mean_squared_error")

    def create_model(self, output_size, activate_function, loss):
        model = models.Sequential([
            layers.Dense(100, input_dim=self.INPUTDIM, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(200, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(output_size, activation=activate_function)
        ])
        model.compile(loss=loss, optimizer=optimizers.Adam(0.001))
        return model

    def save_model(self, file_path='CartPole-v0-PG_baseline.h5'):
        print('model saved')
        self.PG_NET.save(file_path)

    def remember(self, s, a, r):
        self.trajectory.append((s, a, r))

    def train(self, gamma=0.95):
        # X=state, Y=action prob
        df = pd.DataFrame(self.trajectory, columns=['observation', 'action', 'reward'])
        x = np.stack(df['observation'])

        # 计算带折扣的reward之和=G
        df['discount'] = gamma ** df.index.to_series()
        df['discounted_reward'] = df['discount'] * df['reward']
        df['discounted_return'] = df['discounted_reward'][::-1].cumsum()
        # 储存一次该状态下的 reward 汇报 G，
        df['psi'] = df['discounted_return']
        # 计算baseline， 即每一个状态的价值函数，通过神经网络，输入state，输出价值函数 
        df['baseline'] = self.BASELINE.predict(x)
        df['return'] = df['discounted_return'] / df['discount']
        y = df['return'].values[:, np.newaxis]
        self.BASELINE.fit(x, y, verbose=0)
        # 更新策略函数
        df['psi'] = df['psi'] - (df['baseline'] * df['discount'])
        y = np.eye(self.OUTPUTDIM)[df['action']] * df['psi'].values[:, np.newaxis]
        self.PG_NET.fit(x, y, verbose=0)

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