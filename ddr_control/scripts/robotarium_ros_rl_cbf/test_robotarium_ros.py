from numpy.random import gamma

from rcbf_sac.cbf_qp import CascadeCBFLayer
import numpy as np
from matplotlib import patches

from rps.examples.Simple_Reach import Simple_Catcher
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
from rl_ros_robotarium_env import RobotariumEnv

# N = 2
N = 5

a = np.array([[-1.5, -0.5, 0]]).T
initial_conditions = a
for idx in range(1, N):
    initial_conditions = np.concatenate((initial_conditions, np.array(
        [[2 * np.random.rand() - 1.0, 2. * np.random.rand() - 1.0, 6.28 * np.random.rand() - 3.14]]).T), axis=1)


# print('Initial conditions:')
# print(initial_conditions)


# from http://www.sharejs.com
class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()


args = DottableDict()
args.allowDotting()
args.hazards_radius = 0.3
args.hazards_locations = np.array([[0., 0.], [-1., 1.], [-1., -1.]])
args.u_min = np.array([-1, -1])
args.u_max = np.array([1, 1])

cbflayer = CascadeCBFLayer(args)

env = RobotariumEnv()
i = 0
times = 0

time.sleep(3)
env.reset()
while (True):
    x = env.get_pose().T

    # Observe & Predict
    u_norm = Simple_Catcher(env.goal_pos, x[0])

    enemy_x = np.delete(x[1:], 2, 1)
    # print(enemy_x)
    other_s = np.vstack((enemy_x, args.hazards_locations))
    u_safe = cbflayer.get_u_safe(u_norm, x[0], other_s)
    u_safe = u_norm + u_safe
    obs, other_s, reward, done, info = env.step(u_safe)
    if done:
        break
