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

        # no such object qfrc....
        # print(self.sim.data.qfrc_actuation)
        # try qfrc_actuator
        # print(self.sim.data.qfrc_actuator)

        reward = [] # Type: List[float]
        # This term was meant to minize actuator power, but I've since 
        # changed how the actuators work. Instead of commanding force/torque,
        # we are now giving the actuators position commands. This means
        # we should no longer minimize the actuator commands.

        # Reward for smooth transitions
        #if self.paction is not None:
        #    reward.append(- 0.87 * np.absolute((action - self.paction) / self.dt).sum())
        #reward.append(np.square(self.sim.data.qfrc_actuator).sum())

        # Reward for changing the angle (make it spin)
        #reward.append(17000*(new_pos[2] - prev_pos[2])/self.dt)
        # TODO: make the reward control the square of the difference between the action and the joint angles!
        # this would be an estimate of the torque output from the position actuators

        # Penalizes when the legs have different tang position
        # reward.append(-1 * abs(new_pos[3] - new_pos[5]))
        # reward.append(abs(new_pos[3] - new_pos[5]))

        # X velocity, so it moves forward
        reward.append(1699*(new_pos[0] - prev_pos[0])/self.dt)

        # Penalize the robot for touching the floor 
        # reward.append(-1*(1 / new_pos[1])**2)
        #reward.append(70000*new_pos[1])
        #print(*reward)



        #if new_pos[3] > new_pos[5]:
        #    reward.append((self.sim.data.qfrc_actuator[3] - self.sim.data.qfrc_actuator[1])*10)
        #if new_pos[5] > new_pos[3]:
        #    reward.append((self.sim.data.qfrc_actuator[1] - self.sim.data.qfrc_actuator[3])*10)

        # print(reward_ctrl,reward_run)
        done = False

        self.paction = action
        rewards = {}
        for idx, val in enumerate(reward):
            rewards[idx] = val
        return ob, sum(reward), done, rewards

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        self.paction = None
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
