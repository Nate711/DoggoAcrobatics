#### PUT THIS FILE IN gym/gym/envs/mujoco/ !!! #######


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.paction = 0
        self.attr = {}
        self.attr['prev_leg_v_f'] = 0
        self.attr['prev_leg_v_b'] = 0
        # self.attr['leg_cycle'] = 0
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
                reward[str(i) + '_torso_contact'] = -1242

        # Penalize every time the leg change direction to avoid jerking 
        f_v = (new_pos[4] - prev_pos[4])/self.dt
        b_v = (new_pos[6] - prev_pos[6])/self.dt
        if (self.attr['prev_leg_v_f'] > 0 and f_v < 0) or (self.attr['prev_leg_v_f'] < 0 and f_v > 0):
            reward['jerking_f'] = -100
            # self.attr['leg_cycle'] += 1
        if (self.attr['prev_leg_v_b'] > 0 and b_v < 0) or (self.attr['prev_leg_v_b'] < 0 and b_v > 0):
            reward['jerking_b'] = -100
            # self.attr['leg_cycle'] += 1
        self.attr['prev_leg_v_b'] = b_v
        self.attr['prev_leg_v_f'] = f_v
        # reward['jerk'] = -5*self.attr['leg_cycle']

        # Reward for smooth transitions
        # had 1 before
        reward['smooth_transition'] = - (1e-2) * np.absolute(action - self.paction).sum()/self.dt
        # if reward['smooth_transition'] > 10:
        #     print('TOOOOO MCUUUUUCH')
        # reward.append(np.square(self.sim.data.qfrc_actuator).sum())
        
        # Reward for changing the angle (make it spin)
        reward['angular velocity'] = 35*(new_pos[2] - prev_pos[2])/self.dt
        # if new_pos[2] < 0.1:
        #     reward['angular position'] = 6500*new_pos[2]
        # elif new_pos[2] < 1:
        #     reward['angular position'] = 650*new_pos[2]      
        # elif new_pos[2] < 10:
        #     reward['angular position'] = 65*new_pos[2]
        # elif new_pos[2] < 100:
        #     reward['angular position'] = 6*new_pos[2]
        # else:
        #     reward['angular position'] = 0.6*new_pos[2]

        # X velocity, so it moves forward
        reward['x velocity'] = 85*(new_pos[0] - prev_pos[0])/self.dt 

        # Penalize the robot for touching the floor 
        reward['y position'] = 7*(new_pos[1] + 10)
        # reward['y velocity'] = abs(50*(new_pos[1] - prev_pos[1])/self.dt)

        done = False

        def print_values():
            print('\n\nNEW')
            for idx, val in reward.items():
                print('{}: {}'.format(idx, val))

        # print_values()

        self.paction = action
        return ob, sum(reward.values()), done, reward

    def _get_obs(self):
        r = self.sim.data.qpos[2] % 3.14
        r = 1.0 if r < 3.14 else -1.0
        arr = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])
        return np.append(arr, [r])
        # return np.concatenate([
        #     self.sim.data.qpos.flat[1:],
        #     self.sim.data.qvel.flat,
        # ])

    def reset_model(self):
        self.paction = 0
        self.attr = {}
        self.attr['prev_leg_v_f'] = 0
        self.attr['prev_leg_v_b'] = 0
        # self.attr['leg_cycle'] = 0
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.paction = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
