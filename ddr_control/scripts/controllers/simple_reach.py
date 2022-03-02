#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========IEEE_TAI_DDR===============
@File: simple_reach:简单的到达目标点的控制
@Time: 2021/11/14 上午10:29
@Author: chh3213
========Above the sun, full of fire!=============
"""

import numpy as np


def simple_reach(obs, target):
    """
    简单的控制：agent径直往目标点去
    :param obs: 单个智能体obs
    :param target: 该智能体的目标点
    :return: 该智能体的action
    """
    pos = np.zeros(3)
    pos[0] = target[0] - obs[0]  # delta_x
    pos[1] = target[1] - obs[1]  # delta_y
    pos[2] = obs[2]  # 欧拉角
    ct = np.sqrt(np.square(pos[0]) + np.square(pos[1]))
    ct = pos[0] / ct
    theta = np.arccos(ct)
    if pos[1] > 0:
        theta = theta
    elif pos[1] <= 0:
        theta = -theta
    pos[2] = theta - pos[2]
    if pos[2] > 3.142:
        pos[2] -= 6.28
    elif pos[2] < -3.142:
        pos[2] += 6.28
    if pos[2] < 0:
        # pos[2] = -2
        '''越对正，角速度越小'''
        pos[2] = 2 * np.sin(pos[2])
    if pos[2] > 0:
        # pos[2] = 2
        '''越对正，角速度越小'''
        pos[2] = 2 * np.sin(pos[2])
    if pos[2] == 0:
        pos[2] = 0
    action = -pos[2]
    return action
