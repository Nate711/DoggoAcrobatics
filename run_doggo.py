import gym
import time
import numpy as np


# Call the parser before simulation to make sure the xml is up to date!
from subprocess import call
call(["python3", "xml_parser.py"])

env = gym.make('HalfCheetah-v2')
env.reset()

tf = 3
dt = 0.0001
env.render()

# Actions are 4-vectors: one scalar for each joint torque
N = int(tf/dt)
height = np.zeros((2,N))

pitch = 0
for i in range(N):
	
	# no_action = np.zeros((4,))
	# env.step(no_action)

	# jump = np.array([-2,2,-2,2])
	# env.step(jump)

	random_action = env.action_space.sample()
	# env.step(random_action)
	# print(random_action)
	if i*dt > 1:

		if pitch < 1.65:
			action = np.array([-.04, 0 , 0.05, -pitch])
		elif pitch < 3.14:
			action = np.array([-0.11, 0, -0.11, -pitch])
		else:
			action = np.array([0.00, 0, 0, 0])
	else:
		action = np.array([0.05, 0 , 0.05, 0])
	observation, reward, done, info = env.step(action)
	height[:,i] = observation[0:2]

	pitch = (observation[1])

	env.render()

	# time.sleep(0.0005)


print(height[0,:])
origin_y = 0.5
max_height = max(height[0,:]) + origin_y

print('Max height: {}'.format(max_height))
	# time.sleep(0.01)

	# input("Press Enter")
	# break