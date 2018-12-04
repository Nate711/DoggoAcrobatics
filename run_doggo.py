import gym
import time
import numpy as np


# Call the parser before simulation to make sure the xml is up to date!
from subprocess import call
call(["python3", "xml_parser.py"])

env = gym.make('HalfCheetah-v2')
env.reset()

tf = 3
dt = 0.005 # 0.001
env.render()

# Actions are 4-vectors: one scalar for each joint torque
N = int(tf/dt)
height = np.zeros((2,N))

pitch = 0

MAX_UP = 0.057
MAX_DOWN = -0.112

## qpos and action
#index: [0      1       2       3       4       5       6]
#qpos:  [x,     y,      pitch,  ftan,   frad,   btan,   brad]
#act:   [ftan,  frad,   btan,   brad]

## observation
# pitch and joint pos + derivatives

for i in range(N):


	random_action = env.action_space.sample()

	if i*dt > 1:

		if pitch < 1.9:
			action = np.array([-pitch, MAX_UP, 0, MAX_DOWN*1.0]) #100 means maximum
		elif pitch < 2.2:
			action = np.array([-pitch, MAX_DOWN, 0, 0])
		else:
			action = np.array([0, 0, 2*3.14-pitch, 0])
	else:
		action = np.array([0 ,0.08, 0, 0.08])

	observation, reward, done, info = env.step(action)
	height[:,i] = observation[0:2]

	pitch = (observation[0])


	env.render()

	# time.sleep(0.0005)


print(height[0,:])
origin_y = 0.5
max_height = max(height[0,:]) + origin_y

print('Max height: {}'.format(max_height))
	# time.sleep(0.01)

	# input("Press Enter")
	# break