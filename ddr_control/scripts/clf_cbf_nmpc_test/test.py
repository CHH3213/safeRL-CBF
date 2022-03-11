#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========safe-rl-cbf===============
@File: example.py
@Time: 2022/3/7 下午10:36
@Author: chh3213
@Description: 测试env正不正常
========Above the sun, full of fire!=============
"""
import sys
import random
import time
import numpy as np
from observer import Observer
from estimator import Estimator
from clf_cbf_nmpc import CLF_CBF_NMPC, Simple_Catcher
import os

np.set_printoptions(suppress=True)  # 取消科学计数法输出
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))

from ros_robotarium_env import Robotarium_env
from simple_pv import simple_pursuit


def is_done(x):
    """
    判断是否完成目标、被碰撞、智能体是否出界
    """
    self_state = x[0]
    other_states = x[1:]

    # Check boundaries
    # if (self_state[1] > 1.5 or self_state[1] < -1.5 or self_state[0] > 1.5 or self_state[0] < -1.5):
    #     print('Out of boundaries !!')
    #     return True
    # Reached goal?
    if (1.9 <= self_state[0] <= 2.1 and 1.9 <= self_state[1] <= 2.1):
        print('Reach goal successfully!')
        return True

    for idx in range(np.size(other_states, 0)):
        # if(other_states[idx][0]>1.5 or other_states[idx][0]<-1.5 or other_states[idx][1]>1.5 or other_states[idx][1]<-1.5 ):
        #     print('Vehicle %d is out of boundaries !!' % idx+1)
        #     return True
        distSqr = (self_state[0] - other_states[idx][0]) ** 2 + (self_state[1] - other_states[idx][1]) ** 2
        if distSqr < (0.2) ** 2:
            print('Get caught, mission failed !')
            return True

    return False


if __name__ == '__main__':
    env = Robotarium_env()
    env.reset()

    x = env.get_pose().T
    i = 0
    times = 0

    obsrvr = Observer(x, 0.1, 6)

    mpc_horizon = 10
    T = 0.1
    m_cbf = 8
    m_clf = 0
    gamma_k = 0.25
    alpha_k = 0.1
    clf_cbf_nmpc_solver = CLF_CBF_NMPC(mpc_horizon, T, m_cbf, m_clf, gamma_k, alpha_k)
    while (not is_done(x)):
        # print(is_done(x))
        # print('\n----------------------------------------------------------')
        # print("Iteration %d" % times)

        x = env.get_pose().T

        # Observe & Predict

        obsrvr.feed(x)
        f = lambda x_, u_: x_ - x_ + u_
        # print(obsrvr.vel[1:])
        estmtr = Estimator(x[1:], obsrvr.vel[1:], f, 0.1, 10)
        estmtr.predict()
        # print(estmtr.predict_states)
        global_states_sol, controls_sol, local_states_sol = clf_cbf_nmpc_solver.solve(x[0], env.target,
                                                                                      np.concatenate(
                                                                                          (
                                                                                              np.array(
                                                                                                  [obsrvr.states[1:]]),
                                                                                              estmtr.predict_states),
                                                                                          axis=0))
        attacker_u = controls_sol[0]
        # attacker_u = np.array([0.2, 0.1])
        print(attacker_u)
        # defender_u = Simple_Catcher(x[0],x[1])

        dxu = np.zeros([env.agent_number, 2])
        dxu[0] = np.array([attacker_u[0], attacker_u[1]])

        for idx in range(1, env.agent_number):
            # defender_u = Simple_Catcher(x[0],x[idx])
            # dxu[idx] = defender_u
            # dxu[idx] = np.array([0, 0])
            # dxu[idx] = np.array([0.15, 0.1])
            # dxu[idx] = np.array([0.15, 0.1])
            # for idx in range(3, N)
            # defender_u = Simple_Catcher(x[0], x[idx])
            # dxu[idx] = defender_u
            #     dxu[idx] = np.array([0.2, 0.02])
            u = simple_pursuit(x[idx], x[0])
            dxu[idx] = np.array([0.3, u])
        env._take_action(dxu)
        times += 1
        i += 1
