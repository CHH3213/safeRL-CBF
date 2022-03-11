#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========IEEE_TAI_DDR===============
@File: simple_pv:追击和逃避的控制
@Time: 2021/11/14 下午 21：35
@Author: chh3213
========Above the sun, full of fire!=============
"""
import numpy as np


def simple_pursuit(obs, target_agent):
    """
    故障智能体变为追击离他最近的智能体
    :param obs: 该智能体obs
    :param target_agent: 该智能体出故障后追击的目标agent
    :return: 该智能体的action
    """

    pos = np.zeros(3)
    pos[0] = target_agent[0] - obs[0]  # delta_x
    pos[1] = target_agent[1] - obs[1]  # delta_y
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
        pos[2] = 2 * np.sin(pos[2])
    if pos[2] > 0:
        pos[2] = 2 * np.sin(pos[2])
    if pos[2] == 0:
        pos[2] = 0
    # action = -pos[2]  # big size
    action = pos[2]  # nanorobot
    return action


def simple_avoid(obs, pursuer_obs):
    """
    单纯地躲避追击而来的智能体
    :param pursuer_obs: 追击者的状态信息
    :param obs: 自身的状态信息
    :return: 躲避被追击的action
    """
    pos = [0, 0, 0]
    pos[0] = pursuer_obs[0] - obs[0]  # delta_x
    pos[1] = pursuer_obs[0] - obs[0]  # delta_y
    pos[2] = obs[2]  # 欧拉角
    ct = np.sqrt(np.square(pos[0]) + np.square((pos[1])))
    cost = pos[0] / ct
    theta = np.arccos(cost)  # 相对位置夹角
    if pos[1] > 0:
        theta = -theta
    elif pos[1] <= 0:
        theta = theta
    pos[2] = theta - pos[2]
    if 0 < pos[2] < 3.14 / 2:
        pos[2] -= 3.14 - theta
    elif 0 > pos[2] > -3.14 / 2:
        pos[2] += 3.14 + theta
    if pos[2] < 0:
        pos[2] = -2
    if pos[2] > 0:
        pos[2] = 2
    if pos[2] == 0:
        pos[2] = 0
    if ct < 5:
        action = -pos[2]
    return action
