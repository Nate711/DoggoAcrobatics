import gym
import time
import numpy as np


# Call the parser before simulation to make sure the xml is up to date!
from subprocess import call
call(["python3", "xml_parser.py"])

env = gym.make('HalfCheetah-v2')
env.reset()

tf = 3
dt = 0.005
env.render()

# Actions are 4-vectors: one scalar for each joint torque
N = int(tf/dt)
height = np.zeros((2,N))

pitch = 0

MAX_UP = 0.057
MAX_DOWN = -0.112

for i in range(N):
	
	# no_action = np.zeros((4,))
	# env.step(no_action)

	# jump = np.array([-2,2,-2,2])
	# env.step(jump)

	random_action = env.action_space.sample()
	# env.step(random_action)
	# print(random_action)
	if i*dt > 1:

		if pitch < 1.7:
			action = np.array([MAX_DOWN*1.0, 0 , MAX_UP, -pitch]) #100 means maximum
		elif pitch < 2.2:
			action = np.array([0, 0, MAX_DOWN, -pitch])
		else:	
			action = np.array([0, 2*3.14-pitch, 0, 0])
	else:
		action = np.array([0.08, 0 , 0.08, 0])

	observation, reward, done, info = env.step(action)
	height[:,i] = observation[0:2]

	pitch = (observation[1])

	# print(observation)

	env.render()

	# time.sleep(0.0005)


print(height[0,:])
origin_y = 0.5
max_height = max(height[0,:]) + origin_y

print('Max height: {}'.format(max_height))
	# time.sleep(0.01)

	# input("Press Enter")
	# break