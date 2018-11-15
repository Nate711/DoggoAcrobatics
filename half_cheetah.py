#### PUT THIS FILE IN gym/gym/envs/mujoco/ !!! #######


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.paction = 0
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        # mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        prev_pos = (self.sim.data.qpos).copy()
    
        self.do_simulation(action, self.frame_skip)

        new_pos = (self.sim.data.qpos).copy()

        # print(np.absolute(new_pos - prev_pos).sum())

        ob = self._get_obs()

        reward = [] # Type: List[float]
        

        # Reward for smooth transitions
        # had 1 before
        reward.append(- (1e-5) * np.absolute(action - self.paction).sum()/self.dt) # -> becomes 5 is m/s is 1
        #reward.append(np.square(self.sim.data.qfrc_actuator).sum())

        # Reward for changing the angle (make it spin)
        reward.append(1.0*(new_pos[2] - prev_pos[2])/self.dt) # -> becomes 10 if rad/s is 1

        # Penalizes when the legs have different tang position
        # reward.append(-1 * abs(new_pos[3] - new_pos[5]))

        # X velocity, so it moves forward
        reward.append(0.0*(new_pos[0] - prev_pos[0])/self.dt) # -> becomes 0.1 if m/s is 1

        # Penalize the robot for touching the floor 
        # reward.append(-(1 / new_pos[1])**2)

        done = False

        self.paction = action
        rewards = {}
        for idx, val in enumerate(reward):
            rewards[idx] = val
        return ob, sum(reward), done, rewards

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:3],
            self.sim.data.qvel.flat[0:3],
        ])
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.paction = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
