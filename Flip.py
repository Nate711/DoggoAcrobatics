import gym
import numpy as np
import PupperXMLParser
import UpdateGymFiles

env = gym.make("HalfCheetah-v2")
env.reset()

# Duration of simulation
tf = 10

# Tiemstep
dt = 0.005

# Total number of timesteps
N = int(tf / dt)

# Set up the scene
env.render()

# Keep track of the robot's pitch for timing the flip
pitch = 0

# Leg extensions
MAX_UP = 0.10
MAX_DOWN = -0.125

# Actions are 12-vectors. Check out the top of pupper.xml for documentation
action = np.zeros(12)

# Keep a history of the robot's body position
position_history = np.zeros((3, N))

for i in range(N):
    if i * dt > 2:
        if pitch < 0.9:
            action = np.array([0, 0, MAX_DOWN] * 2 + [0, pitch, MAX_UP] * 2)  # 100 means maximum
        elif pitch < 2.2:
            action = np.array([0, pitch, MAX_DOWN]*4)
        else:
            action = np.array([0, 0, 0] * 4)
    else:
        action = np.array([0, 0, MAX_UP]*4)

    # Key line: applies the action we decided on (servo positions) and
    # steps the simulation forward one timestep
    observation, reward, done, info = env.step(action)
    position_history[:, i] = observation[0:3]

    quat_orientation = observation[3:7]
    w = quat_orientation[0]
    angle = 2 * np.arccos(w)
    vector = quat_orientation[1:] / (1 - w * w) ** 0.5
    pitch = angle
    env.render()
