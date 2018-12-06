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
        ANG_VEL = 0
        YPOS = 10
        Y_THRESHOLD = 0.1 # [m]
        GROUND = 10


        TORQUE_NORMALIZER = [1,0.02,1,0.02] # puts torque (-5 to 5) and linear force (-250 to 250) on the same magnitude
        TORQUE_GAIN = 1e-1 # multiplies both torque and normalized linear force

        prev_pos = (self.sim.data.qpos).copy()
        self.do_simulation(action, self.frame_skip)
        new_pos = (self.sim.data.qpos).copy()

        body_pitch_old = prev_pos[2]
        body_pitch_new = new_pos[2]
        body_y = new_pos[1]

        full_state = (self.sim.data.qpos).copy()

        ob = self._get_obs()
        reward = {} # Type: List[float]

        ## Limit force
        #index: [0      1       2       3       4       5       6]
        #qpos:  [x,     y,      pitch,  ftan,   frad,   btan,   brad]
        #act:   [ftan,  frad,   btan,   brad]

        leg_positions = new_pos.take([3,4,5,6])
        torque_estimate = (action - leg_positions)*TORQUE_NORMALIZER # element-wise multiplication
        torque_estimate = np.square(torque_estimate).sum()

        reward['torque minimizer'] =  - torque_estimate * TORQUE_GAIN

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c1 = self.sim.model.geom_id2name(contact.geom1)
            c2 = self.sim.model.geom_id2name(contact.geom2)
            
            if c2 == 'torso':
                reward[c1 + '_torso_contact'] = -GROUND
	    
        # if reward['smooth_transition'] > 10:
        #     print('TOOOOO MCUUUUUCH')
        
        # Reward for changing the angle (make it spin)
        reward['angular velocity'] = ANG_VEL * (body_pitch_new - body_pitch_old)/self.dt

        # Reward the robot for being more thna y_threshold off the ground
        reward['y position'] = YPOS * ((body_y - Y_THRESHOLD)**2) if body_y > Y_THRESHOLD else 0

        done = False

        def print_values():
            print('\n\nNEW')
            for idx, val in reward.items():
                print('{}: {}'.format(idx, val))

        # print_values()

        self.paction = action
        return ob, sum(reward.values()), done, reward

    def _get_obs(self):
        # take [pitch, joint positions] + derivative wrt time of [pitch, joint positions]
        on_ground = 0
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c1 = self.sim.model.geom_id2name(contact.geom1)
            c2 = self.sim.model.geom_id2name(contact.geom2)
            if c2 == 'fradial' or c2 == 'bradial':
                 on_ground = 1 

        return np.concatenate([
            np.array(self.sim.data.qpos.flat).take([1,2,3,4,5,6]),
            np.array(self.sim.data.qvel.flat).take([1,2,3,4,5,6]),
            np.array([on_ground])
        ])

    def reset_model(self):
        self.paction = 0
        self.attr = {}
        self.attr['prev_leg_v_f'] = 0
        self.attr['prev_leg_v_b'] = 0
        # self.attr['leg_cycle'] = 0
        qpos = self.init_qpos + self.np_random.uniform(low=-.001, high=.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .001
        self.paction = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
