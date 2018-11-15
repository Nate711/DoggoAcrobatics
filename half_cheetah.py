#### PUT THIS FILE IN gym/gym/envs/mujoco/ !!! #######


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        # mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        anglebefore = self.sim.data.qpos[2]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        angleafter = self.sim.data.qpos[2]
        ob = self._get_obs()

        # no such object qfrc....
        # print(self.sim.data.qfrc_actuation)
        # try qfrc_actuator
        # print(self.sim.data.qfrc_actuator)


        # This term was meant to minize actuator power, but I've since 
        # changed how the actuators work. Instead of commanding force/torque,
        # we are now giving the actuators position commands. This means
        # we should no longer minimize the actuator commands.
        # reward_ctrl = - 0.1 * np.square(action).sum()
        reward_ctrl = 0.001 * np.square(self.sim.data.qfrc_actuator).sum()
        reward_spin = 100*(angleafter - anglebefore)/self.dt 
        # TODO: make the reward control the square of the difference between the action and the joint angles!
        # this would be an estimate of the torque output from the position actuators


        reward_run = (xposafter - xposbefore)/self.dt

        # print(reward_ctrl,reward_run)

        reward = reward_ctrl + reward_run + reward_spin 
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
