# This file must at some point be copied to the gym/gym/envs/mujoco/ directory

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    POMDP model for the Pupper robot
    Modified from the HalfCheetah model to work with OpenAI Gym
    """

    def __init__(self):
        # previous action
        self.paction = 0
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 1)
        # mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        """
        Execute the given action and then compute the resulting reward.

        :param action: The action to take.
        """
        self.do_simulation(action, self.frame_skip)
        new_pos = (self.sim.data.qpos).copy()
        ob = self._get_obs()

        # Extract the y-axis rate of the robot
        pitch_rate = ob[23]
        vel_x = ob[19]

        reward = {}  # Type: List[float]

        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            c1 = self.sim.model.geom_id2name(contact.geom1)
            c2 = self.sim.model.geom_id2name(contact.geom2)
            if c2 == "torso":
                reward[str(i) + "_torso_contact"] = -531

        # Penalizes big changes in servo positions
        reward["smooth_transition"] = (
            -(1e-2) * np.absolute(action - self.paction).sum() / self.dt
        )

        # Reward for spinning
        reward["angular velocity"] = 35 * pitch_rate

        # Reward the x velocity, so that it moves forward
        reward["x velocity"] = 85 * vel_x

        # Penalize the robot for touching the floor
        reward["y position"] = 7 * new_pos[1]

        done = False

        def print_values():
            print("\n\nNEW")
            for idx, val in reward.items():
                print("{}: {}".format(idx, val))

        # print_values()

        self.paction = action
        return ob, sum(reward.values()), done, reward

    def _get_obs(self):
        arr = np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
        return arr

    def reset_model(self):
        self.paction = 0
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.01
        self.paction = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
