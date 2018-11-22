#### PUT THIS FILE IN gym/gym/envs/mujoco/ !!! #######


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.paction = None
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        # mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        prev_pos = (self.sim.data.qpos).copy()
    
        self.do_simulation(action, self.frame_skip)

        new_pos = (self.sim.data.qpos).copy()

        ob = self._get_obs()

        reward = {} # Type: List[float]

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c1 = self.sim.model.geom_id2name(contact.geom1)
            c2 = self.sim.model.geom_id2name(contact.geom2)
            # print('contact', i)
            # print('dist', contact.dist)
            # print('geom1', contact.geom1, self.sim.model.geom_id2name(contact.geom1))
            # print('geom2', contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            if c2 == 'torso':
                reward[str(i) + '_torso_contact'] = -100

        # TODO: This doesn't really work. With weight 100 angular velocity makes everything awful
        #       with weight 10 it's being 'ignored' which means it's slightly worse than weight 0
        #       Maybe changing the weight to something in between could work.
        #       Note that without angular velocity it's just doing shitty jumps to the side.
        #       Maybe improving the shitty jumps first could be useful.

        # Reward for smooth transitions
        # had 1 before
        reward['smooth_transition'] = - (1e-5) * np.absolute(action - self.paction).sum()/self.dt
        # reward.append(np.square(self.sim.data.qfrc_actuator).sum())
        
        # Reward for changing the angle (make it spin)
        reward['angular velocity'] = 10*(new_pos[2] - prev_pos[2])/self.dt

        # X velocity, so it moves forward
        reward['x velocity'] = 85*(new_pos[0] - prev_pos[0])/self.dt 

        # Penalize the robot for touching the floor 
        reward['y position'] = 3.7*(new_pos[1] + 10)
        reward['y velocity'] = abs(50*(new_pos[1] - prev_pos[1])/self.dt)

        done = False

        def print_values():
            print('\n\nNEW')
            for idx, val in reward.items():
                print('{}: {}'.format(idx, val))

        # print_values()

        self.paction = action
        return ob, sum(reward.values()), done, reward

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])

    def reset_model(self):
        self.paction = None
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.paction = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
