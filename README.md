# DoggoAcrobatics

## Requirements
Follow the OpenAI gym installation instructions. Make sure to also install Mujoco using a student license / 30 day trial license.

1. Modify xml_parser.py so that envs_path is your ```gym/envs/``` folder.

2. Execute the following to run the simulator:
```
python3 run_doggo.py
```
3. Execut the following to train doggo:
```
bash train.sh
```
4. Use ```python -m spinup.run plot Working_results/train_doggo4_s0/``` to plot and ```python -m spinup.run test_policy Working_results/train_doggo4_s0/``` to simulate.