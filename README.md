# Pupper RL
## Installation
0. Follow the OpenAI Spinning Up installation instructions: (https://spinningup.openai.com/en/latest/user/installation.html). Make sure to copy your mujoco mjkey.txt file to the right location inside
the mujoco-py directory.

1. Modify UpdateGymFiles.py so that ENVS_PATH is correctly set to your ```gym/envs/``` folder.

2. To test the simulator and have the robot do a backflip run:
```console
python3 Flip.py
```

## RL Parameters
The file ```half_cheetah.py``` is the brains of the robot. It's responsible for:
1. Taking a partial observation of the environment
2. Determining the action
3. Calculating the reward of that action

Since I updated the project to use the Pupper model, I'm not sure if the reward function we implemented still works.

## Training with Spinning Up
1. Run this bash script to train Pupper using the Spinning Up library
```console
bash train.sh
```
You'll need a beefy computer to train the model with any speed.
2. Run
```console
python -m spinup.run plot Working_results/train_doggo4_s0/
```
to plot the training results.
2. Run
```console
python -m spinup.run test_policy Working_results/train_doggo4_s0/
```
to simulate the trained policy.
