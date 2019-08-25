# DoggoAcrobatics

## Requirements
0. Follow the OpenAI Spinning Up installation instructions: (https://spinningup.openai.com/en/latest/user/installation.html). Make sure to copy your mujoco mjkey.txt file to the right location inside
the mujoco-py directory.

1. Modify UpdateGymFiles.py so that ENVS_PATH is correctly set to your ```gym/envs/``` folder.

2. To test the simulator and have the robot do a backflip run:
```console
python3 Flip.py
```
3. Run this bash script to train doggo using the Spinning Up library
```console
bash train.sh
```
4. Run
```console
python -m spinup.run plot Working_results/train_doggo4_s0/
```
to plot the training results.
5. Run
```console
python -m spinup.run test_policy Working_results/train_doggo4_s0/
```
to simulate the trained policy.
