import math
import numpy as np
import matplotlib.pyplot as plt

import random
from random import gauss

class Kbandits(object):
	'''
		k-臂老虎机，初始化：
			k个老虎机的真实价值 q*(At) 服从是均值是0，方差是1 的正太分布
			他们的实际收益 rt 是 q*(At) 为均值，方差为1 的正太分布
	'''
	def __init__(self, k):
		mean = 0
		variance = 1
		self.qat = [gauss(mean, math.sqrt(variance)) for i in range(int(k))]
		self.eps = 0.01

	def gen_rewards(self, k):
		qat = self.qat[k]  # mean
		variance = 1         # variance
		reward = gauss(qat, math.sqrt(variance))
		return reward

	def egreedy(self, rande, Q):
		if rande < Kbandits.eps:
			k = list(np.random.randint(K, size=1))[0]
		else:
			k = np.argmax(Q)
		return k

	def greedy(self, Q):
		k = np.argmax(Q)
		return k

	def UCB(self, Q, count, t):
		R = []
		c = 2
		for q in range(len(Q)):
			if count[q] == 0:
				count[q] = count[q] + 1
				return q
			else:
				r = Q[q]+c*(math.sqrt(math.log(t, 2.71828)/count[q]))
				R.append(r)
				print(t, r)
		return np.argmax(R)

T = 1000  # 迭代次数
K = 10    # K - 臂老虎机
Kbandits = Kbandits(K)
r = 0
r_all = []
Q = list([0]*K)
count = list([0]*K)

for t in range(1,T+1):
	rande = random.random()
	# algorithm
	# k = Kbandits.egreedy(rande, Q)  # K: egreedy
	# k = Kbandits.egreedy(Q)         # K: greedy
	k = Kbandits.UCB(Q, count, t) 
	v = Kbandits.gen_rewards(k)
	r = (r*t + v)/(t+1)
	r_all.append(r) # 画图
	Q[k] = (Q[k]*count[k] + v)/(count[k]+1)
	count[k] = count[k] + 1

plt.plot(list(range(T)),r_all,color='r')
plt.show()
