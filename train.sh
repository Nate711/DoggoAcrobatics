python xml_parser.py ; python -m spinup.run ppo --hid "[128, 128]" --env HalfCheetah-v2 --exp_name train_doggo5 --gamma 0.99 --max_ep_len 4000 --steps_per_epoch 8000 --epochs 1000 2>&1 | tee output.txt

