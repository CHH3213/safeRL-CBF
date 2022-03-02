# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
==========IEEE_TAI_DDR===============
@File: real_env_test:测试走完整个来回
@Time: 2021/11/26 下午15:16
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
from envs.motion_capture import Motion_capture
from envs.control_car import DDR
import random
from controllers.CBF import CBF
from controllers.simple_reach import simple_reach


# Twist - 线速度角速度
# pose - 位置：position为实际位置，
class TAI_Env:
    def __init__(self):
        self.env_type = 'continuous'
        self.name = 'safe_robots_attribute'
        # 初始化一个node，名字为self.name  anonymous=True，
        # 表示后面定义相同的node名字时候，按照序号进行排列
        # log_level设置日志等级
        # rospy.init_node(self.name, anonymous=True, log_level=rospy.DEBUG)
        rospy.init_node(self.name, anonymous=True)
        # env properties
        # TODO:代码运行频率星需要根据实际情况调整
        self.rate = rospy.Rate(1000)
        self.obs_dict = []  # 存储字典类型obs
        self.agent_number = 3  # agent数量
        self.obs_dim = 13  # 每个agent的observation的维度
        self.act_dim = 1  # action的维度(个数)
        self.observation_space = []
        self.action_space = []
        self.action_space_shape = []
        self.u_range = 1
        self.theta = 0
        self.v_linear = 0.15
        self.safe_radius = 0.51
        # self.action_n = np.array([d, a])
        self.action_n = np.zeros([self.agent_number, 1])
        # TODO:目标点设置
        self.is_origin = [False, False, False]
        self.target = np.array([[0, 0], [2, 2], [2, 2]])
        # TODO：根据算法调整
        for i in range(self.agent_number):
            self.action_space.append(spaces.Box(
                low=-self.u_range, high=+self.u_range, shape=(self.act_dim,), dtype=np.float32))
            self.observation_space.append(spaces.Box(low=np.array([-5, -5, -np.pi, -5, -5, -5, -5, -10, -10, -np.pi, -np.pi, -10, -10]),
                                                     high=np.array([5, 5, np.pi, 5, 5,  5, 5, 10, 10, np.pi,  np.pi, 10, 10]), dtype=np.float32))
        # robot properties
        self.obs = np.zeros((self.agent_number, self.obs_dim))
        self.prev_obs = np.zeros((self.agent_number, self.obs_dim))
        # ['deactivated','deactivated']
        self.status = ['deactivated'] * self.agent_number

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
        self.state_capture = Motion_capture()

        # positions of walls and barriers setup
        # FIXME:need to according to real world to set
        self.walls = dict(wall_0=np.array([0, 3.1]), wall_1=np.array([0, -3.1]), wall_2=np.array([-3.1, 0]), wall_3=np.array([3.1, 0]))
        self.barriers = dict(barrier_0=np.array([-1.5, 0]), barrier_1=np.array([1.5, 0]), barrier_2=np.array([0,1.5]))

    def reset(self):
        """
        Reset environment
        Usage:
            obs = env.reset()
        """
        # rospy.logdebug("\nStart Environment Reset")
        # set init pose
        self.count_reach = 0
        self.is_origin = [False, False, False]
        self.target = np.array([[0, 0], [2, 2], [2, 2]])
        self._get_obs()
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
        safe_n = []
        #FIXME:TEST WRONG OR FALSE
        for i in range(self.agent_number):
            actions[0] = simple_reach(self.obs[0], self.target[i])
            actions[1] = simple_reach(self.obs[1], self.target[i])

        self._take_action(action_n)
        for i in range(self.agent_number):
            # update status
            reward, done = self._compute_reward(i, self.target[i])
            # self.prev_obs = self.obs.copy() # make sure this happened after reward computing

            done_n.append(done)
            reward_n.append(reward)
            safe_n.append(0)
            self.target[i] = np.array([0, 0]) if self.is_origin[i] else np.array([2, 2])
        if self.count_reach >= self.agent_number:
            done_n = [True] * self.agent_number
        info = self.status
        self._get_obs()

        return self.obs, reward_n, done_n, safe_n, info

    def render(self):
        pass

    def _get_obs(self):
        """
        Returns:
            obs: array([...agent1...,agent2...agent3...])
        一个agent状态包括：
                自己的位置、，自身欧拉角,与其他agent的相对位置，与其他agent的相对距离，相对位置方向夹角.
        对3个智能体的环境而言，每个agent状态应为：2*1+1+2*2+1*2+1*2=11
        """
        obs_all = []  # 储存所有agent状态列表
        obs_dict = []
        positions_all = []
        attitudes_all = []
        for i, ddr_name in enumerate(self.ddr_name):
            # FIXME:state obtained from motion capture
            [model_position, model_attitude, _, _,_] = self.state_capture.get_state(ddr_name)
            positions_all.append(model_position)
            attitudes_all.append(model_attitude[2])  # 只需要欧拉角z
        positions_all = np.array(positions_all)
        attitudes_all = np.array(attitudes_all)
        positions_all = np.delete(positions_all, 2, axis=1)  # 去掉z轴位置
        for i, position in enumerate(positions_all):
            # 对于一个agent来说
            delta_list = positions_all - position
            delta_list = np.delete(delta_list, i, axis=0)
            dist_list = []
            theta_list = []
            for delta in delta_list:
                dist = np.linalg.norm(delta)
                theta = np.arctan(delta[1] / delta[0])
                dist_list.append(dist)
                theta_list.append(theta)
            position = np.array(position).flatten()
            delta_pos = np.array(delta_list).flatten()
            delta_dist = np.array(dist_list).flatten()
            theta_array = np.array(theta_list).flatten()
            # TODO:obs_one_agent 正常的话应该为11维
            obs_one_agent_dict = {'position': position, 'delta_pos': delta_pos,
                                  'delta_dist': delta_dist, 'theta_array': theta_array, 'target': self.target}
            obs_one_agent = np.concatenate((position, [
                                           attitudes_all[i]], delta_pos, delta_dist, theta_array, np.array(self.target[i])-np.array(position)))
            obs_all.append(obs_one_agent)
            obs_dict.append(obs_one_agent_dict)
        self.obs = copy.deepcopy(obs_all)
        self.obs_dict = copy.deepcopy(obs_dict)
        return self.obs

    def __dist2wall_barrier(self, agent_index):
        barrier_positions = []
        dist_agent2barriers = []
        for barrier in self.barrier_name:
            barrier_position = self.barriers[barrier]
            barrier_positions.append(barrier_position)
            dist_agent2barrier = np.linalg.norm(
                np.array(self.obs_dict[agent_index]['position']) - np.array(barrier_position))
            dist_agent2barriers.append(dist_agent2barrier)
        barrier_pos = barrier_positions[np.argmin(dist_agent2barriers)]

        dist_agent2walls = []
        wall_positions = []
        for wall in self.walls_name:
            wall_position = self.walls[wall]
            if wall == 'wall_0' or wall == 'wall_1':
                dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][1] - wall_position[1])
                wall_position[0] = self.obs_dict[agent_index]['position'][0]
            else:
                dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][0] - wall_position[0])
                wall_position[1] = self.obs_dict[agent_index]['position'][1]

            dist_agent2walls.append(dist_agent2wall)
            wall_positions.append(wall_position)

        wall_pos = wall_positions[np.argmin(dist_agent2walls)]
        # print('agent_index', agent_index)
        # print(dist_agent2walls)
        # print(wall_positions)
        # print(wall_pos)
        # print('================')
        return barrier_pos, wall_pos
        
    def _take_action(self, actions):
        """
        Publish cmd_vel according to an action index
        Args:
            i_act: array([ia0, ia1])
        Returns:
        """
        # rospy.logdebug("\nStart Taking Action")
        # self.state_capture.unpausePhysics()
        barrier_pos0, wall_pos0 = self.__dist2wall_barrier(agent_index=0)
        barrier_pos1, wall_pos1 = self.__dist2wall_barrier(agent_index=1)

        state1 = np.concatenate((self.obs[2][0:3], self.obs[0][0:3], actions[2],
                                self.obs[1][0:3], actions[1], barrier_pos0[0:2], wall_pos0[0:2]))
        state2 = np.concatenate((self.obs[2][0:3], self.obs[1][0:3], actions[2],
                                self.obs[0][0:3], actions[0], barrier_pos1[0:2], wall_pos1[0:2]))
        state3 = np.concatenate((np.array([10, 10, 10]), self.obs[2][0:3], [
                                0], np.array([10, 10, 10]), [0], barrier_pos1[0:2], wall_pos1[0:2]))
        # print(state2)
        actions[0] = CBF(actions[0], state1)
        actions[1] = CBF(actions[1], state2)
        actions[2] = CBF(-actions[2], state3)
        vel_list = []
        for i, action in enumerate(actions):
            # 角速度为负逆时针转动，为正顺时针转动
            vel = Twist()
            vel.linear.x = self.v_linear
            if i == 2:
                vel.linear.x = self.v_linear*0.8
            vel.angular.z = -action
            vel_list.append(vel)
        ##')

        # print('{} angular.z:{}'.format(self.ddr_name[i], action))

        for _ in range(1):
            for i, vel in enumerate(vel_list):
                self.ddr_list[i].car_vel.publish(vel)
            self.rate.sleep()        # self.state_capture.pausePhysics()
        # rospy.logdebug("\nEnd Taking Action\n")

    def _compute_reward(self, agent_index, target):
        """
        Compute reward and done based on current status
        :param agent_index: 智能体索引
        :param target: 目标点位置(x,y)
        Return:
            reward
            done
        """
        # rospy.logdebug("\nStart Computing Reward")
        # 到目标点的距离
        # print(self.is_origin)
        dist_agent2target = np.linalg.norm(
            np.array(self.obs_dict[agent_index]['position']) - np.array(target))
        reward, done = 0, False
        if dist_agent2target < 0.25 and self.is_origin[agent_index]:
            reward += 300
            if agent_index == 0 or agent_index == 1:
                self.is_origin[agent_index] = False
                self.count_reach += 1
            # done = True
        elif dist_agent2target < 0.25 and not self.is_origin[agent_index]:
            reward += 100
            self.is_origin[agent_index] = True

        reward += -1.0 * dist_agent2target

        # 各个智能体间的距离
        for d in self.obs_dict[agent_index]['delta_dist']:
            if d < 0.25:
                reward -= 1000
                self.is_origin[agent_index] = False
                done = True
            else:
                reward += d
        # rospy.logdebug("\nEnd Computing Reward\n")

        # # 碰到障碍物奖励
        for barrier in self.barrier_name:
            [barrier_position, _, _, _, _] = self.state_capture.get_state(barrier)
            dist_agent2barrier = np.linalg.norm(
                np.array(self.obs_dict[agent_index]['position']) - np.array(barrier_position[0:2]))
            if dist_agent2barrier < 0.25:
                reward -= 1000
                done = True

        # 碰到墙奖励
        for wall in self.walls_name:
            [wall_position, _, _, _, _] = self.state_capture.get_state(wall)
            if wall == 'wall_0' or wall == 'wall_1':
                dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][1] - wall_position[1])
            else:
                dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][0] - wall_position[0])

            if dist_agent2wall < 0.25:
                reward -= 1000
                done = True

        return reward, done


if __name__ == "__main__":
    env = TAI_Env()
    num_steps = int(1e5)
    # obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    # import sys

    # sys.path.append("..")
    from controllers.simple_reach import simple_reach
    from controllers.simple_pv import simple_pursuit, simple_avoid

    for t in range(num_steps):
        # 使用随机action测试
        # actions = random.random(3) - 0.5
        # print('sd', actions)
        '''===========使用简单的到达目标点测试==========='''
        actions = []
        for i, ob in enumerate(o):
            a = simple_reach(ob, env.target[i])
            actions.append(a)
            """==========================================="""
            """当episode大于某个数时，随机一个智能体变为追击离它最近的智能体，相应地智能体也要变为躲避它"""
        if ep >= 1:
            print('start')
            if ep == 1:
                bad_index = random.choice(range(env.agent_number))
                dist = [np.linalg.norm(o[i][0:2] - o[bad_index][0:2])
                        for i in range(env.agent_number)]
                avoid_index = np.argsort(dist)[1]  # 返回第二小的索引

            print('bad_index', bad_index)
            print('avoid_index', avoid_index)
            a_bad = simple_pursuit(o[bad_index], o[avoid_index])
            a_avoid = simple_avoid(o[avoid_index], o[bad_index])
            actions[bad_index], actions[avoid_index] = a_bad, a_avoid
        # print('actions', np.shape(actions))
        o, r, d, i = env.step(actions)

        st += 1
        # rospy.loginfo("\n-\n episode: {}, step: {} \nobs: {}, act: {}\t,"
        #               " reward: {}\t, done: {}\t, info: {}\t"
        #               .format(ep + 1, st, o, actions, r, d, i))
        if any(d):
            # break
            ep += 1
            st = 0
            obs = env.reset()
