# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
==========IEEE_TAI_DDR===============
@File: train_env:
        训练过程中使用，每个智能体到目标点即成功。
        目标点随机，agent初始位置随机
@Time: 2021/11/14 下午15:11
@Author: chh3213
========Above the sun, full of fire!=====
"""
from __future__ import absolute_import, division, print_function
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import sys
import os
import math
import numpy as np
from numpy import pi
import time
import rospy
import tf
import copy
from geometry_msgs.msg import Pose, Twist
from robotarium_get import Robotarium
from control_car import DDR
import random


# Twist - 线速度角速度
# pose - 位置：position为实际位置，
class Robotarium_env:
    def __init__(self):
        self.env_type = 'continuous'
        self.name = 'safe_robots_attribute'
        # 初始化一个node，名字为self.name  anonymous=True，
        # 表示后面定义相同的node名字时候，按照序号进行排列
        # log_level设置日志等级
        # rospy.init_node(self.name, anonymous=True, log_level=rospy.DEBUG)
        rospy.init_node(self.name, anonymous=True)
        # env properties
        # TODO：代码运行频率星需要根据实际情况调整
        self.rate = rospy.Rate(5)
        # TODO：状态动作空间需要再调整
        self.obs_dict = []  # 存储字典类型obs
        self.obs_dim = 13  # 每个agent的observation的维度
        self.act_dim = 1  # action的维度(个数)
        self.agent_number = 5  # agent数量
        self.observation_space = []
        self.action_space = []
        self.action_space_shape = []
        self.u_range = 1
        self.theta = 0
        # self.action_n = np.array([d, a])
        self.action_n = np.zeros([self.agent_number, 1])

        # TODO：根据算法调整
        # 自己的位置、，自身欧拉角, 与其他agent的相对位置，与其他agent的相对距离，相对位置方向夹角.

        for i in range(self.agent_number):
            self.action_space.append(
                spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.act_dim,), dtype=np.float32))
            self.observation_space.append(
                spaces.Box(low=np.array([-5, -5, -np.pi, -5, -5, -5, -5, -10, -10, -np.pi, -np.pi, -10, -10]),
                           high=np.array([5, 5, np.pi, 5, 5, 5, 5, 10, 10, np.pi, np.pi, 10, 10]), dtype=np.float32))
            # self.observation_space.append(spaces.Box(low=np.array([-5, -5, -np.pi, -10, -10]),
            #                                          high=np.array([5, 5, np.pi, 10, 10]), dtype=np.float32))                                         
        # robot properties
        self.obs = np.zeros((self.agent_number, self.obs_dim))
        self.prev_obs = np.zeros((self.agent_number, self.obs_dim))
        self.status = ['deactivated'] * self.agent_number  # ['deactivated','deactivated']

        # barrier名称列表
        self.barrier_name = ['barrier_0', 'barrier_1', 'barrier_2']
        # 墙名称列表
        self.walls_name = ['wall_0', 'wall_1', 'wall_2', 'wall_3']
        # 存储agent列表
        self.ddr_list = []
        self.ddr_name = []
        for i in range(self.agent_number):
            ddr = DDR('ddr_' + str(i))
            self.ddr_list.append(ddr)
            self.ddr_name.append('ddr_' + str(i))
        self.gazebo_reset = Robotarium()
        # TODO:目标点设置
        self.choose_target = np.array([[2, 2, 0], [0, 0, 0]])
        # self.target = random.choice(self.choose_target)
        self.target = self.choose_target[0]
        self.is_success = [False] * self.agent_number
        self.gazebo_reset.reset_target(self.target)

    def reset(self):
        """
        Reset environment
        Usage:
            obs = env.reset()
        """
        # rospy.logdebug("\nStart Environment Reset")
        # set init pose
        self.is_success = [False] * self.agent_number
        # self.target = random.choice(self.choose_target)
        self.target = self.choose_target[0]

        # print('target', self.target)
        self.gazebo_reset.resetWorld()
        self.gazebo_reset.reset_agent_state(ddr_list=self.ddr_list)
        self.gazebo_reset.reset_target(self.target)
        # rospy.logerr("\nEnvironment Reset!!!\n")
        return self.obs

    def step(self, action_n):
        """
        action_n:多个智能体的action
        obs, rew, done, info = env.step(action_n)
        """
        # rospy.logdebug("\nStart environment step")
        # print(action_n)
        done_n = []
        reward_n = []
        self._take_action(action_n)
        for i in range(self.agent_number):
            # update status
            reward, done = self._compute_reward(i, self.target)
            if i == 2:
                reward = 0
            # self.prev_obs = self.obs.copy() # make sure this happened after reward computing
            done_n.append(done)
            reward_n.append(reward)
        # print(self.count_reach)
        if all(self.is_success[0:2]):
            done_n = [True] * (self.agent_number - 1)
            print('done!')
        # print(done_n)
        info = self.status
        self._get_obs()

        # rospy.logdebug("\nEnd environment step\n")
        # print(reward)
        # print(done_n)
        return self.obs, reward_n, done_n, info

    def render(self):
        pass

    def get_pose(self):
        """Returns the states of the agents.

        -> 3xN numpy array (of robot poses)
        """
        poses = np.empty(3)
        for ddr in self.ddr_name:
            [model_position, model_attitude, _, _, _] = self.gazebo_reset.get_state(ddr)
            pose = np.hstack((np.array(model_position[0:2]), np.array(model_attitude[2])))
            temp = copy.deepcopy(pose)
            poses = np.vstack((poses, temp))
        poses = np.delete(poses, 0, 0)  # Nx3
        poses = poses.T  # 3xN
        return poses

    def _take_action(self, actions):
        """
        Publish cmd_vel according to an action index
        Args:
            i_act: array([ia0, ia1])
        Returns:
        """
        # rospy.logdebug("\nStart Taking Action")
        # self.gazebo_reset.unpausePhysics()

        vel_list = []
        for action in enumerate(actions):
            # 角速度为负逆时针转动，为正顺时针转动
            vel = Twist()
            vel.linear.x = action[1][0]
            vel.angular.z = action[1][1]
            vel_list.append(vel)
        # print('********************')
        # print('as', vel_list)
        # print('====================')
        '''==============================='''
        for _ in range(1):
            for i, vel in enumerate(vel_list):
                # print(vel)
                self.ddr_list[i].car_vel.publish(vel)
            self.rate.sleep()
        '''=================================='''
        # self.gazebo_reset.pausePhysics()
        # rospy.logdebug("\nEnd Taking Action\n")
