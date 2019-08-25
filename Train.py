from spinningup.spinup import ppo
import tensorflow as tf
import gym


def env_fn():
    gym.make("HalfCheetah-v2")


ac_kwargs = dict(hidden_sizes=[8, 8], activation=tf.nn.relu)
ppo(
    env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=25, max_ep_len=2000
)
