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
import numpy as np
import time

np.set_printoptions(suppress=True)  # 取消科学计数法输出
sys.path.append('..')
from envs.one_agent_env import TAI_Env

if __name__ == '__main__':
    env = TAI_Env()
    num_steps = int(1e5)
    ep, st = 0, 0
    time.sleep(1)
    o = env.reset()
    # print(np.shape(o))

    for t in range(num_steps):
        # 使用随机action测试
        actions = [random.random()]
        # print('sd', np.shape(actions))
        # print(actions)
        o, r, d, i = env.step(actions)

        st += 1
        # rospy.loginfo("\n-\n episode: {}, step: {} \nobs: {}, act: {}\t,"
        #               " reward: {}\t, done: {}\t, info: {}\t"
        #               .format(ep + 1, st, o, actions, r, d, i))
        if d:
            # break
            ep += 1
            st = 0
            obs = env.reset()
