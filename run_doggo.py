import gym
import time
env = gym.make('HalfCheetah-v2')
env.reset()
import numpy as np

tf = 10
dt = 0.01
env.render()
# input("Press Enter to stop...")
print(env.action_space.sample().shape)

for i in range(int(tf/dt)):
	action = np.ones_like(env.action_space.sample())
	none = np.zeros_like(env.action_space.sample())

	env.step(none)
	env.render()
	time.sleep(0.002)

	# print('fuck you Nathan let\'s use git')
	# print('you;ll thank me later mofo')
