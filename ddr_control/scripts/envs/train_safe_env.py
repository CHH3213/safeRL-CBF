# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
==========IEEE_TAI_DDR===============
@File: train_safe_env:用于safe RL框架
        训练设置：固定一个智能体去追离他最近的智能体，其余两个智能体训练安全地到达目标点
        目标点随机，agent初始位置随机
@Time: 2021/11/17 下午21:10
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

from numpy.lib.utils import safe_eval
import rospy
import tf
import copy
from geometry_msgs.msg import Pose, Twist
from envs.gazebo_reset import Gazebo_reset
from envs.control_car import DDR
import random
from controllers.CBF import CBF


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
        # TODO：代码运行频率星需要根据实际情况调整
        self.rate = rospy.Rate(5)
        # TODO：状态动作空间需要再调整
        self.obs_dict = []  # 存储字典类型obs
        self.obs_dim = 19  # 每个agent的observation的维度
        self.act_dim = 1  # action的维度(个数)
        self.agent_number = 3  # agent数量
        self.observation_space = []
        self.action_space = []
        self.action_space_shape = []
        self.discrete_action_space = False
        self.u_range = 1
        self.theta = 0
        self.v_linear = 0.15
        self.safe_radius = 0.6
        # self.action_n = np.array([d, a])
        self.action_n = np.zeros([self.agent_number, 1])

        # TODO：根据算法调整
        # 自己的位置、，自身欧拉角, 与其他agent的相对位置，与其他agent的相对距离，相对位置方向夹角.

        for i in range(self.agent_number):
            self.action_space.append(
                spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.act_dim,), dtype=np.float32))
            self.observation_space.append(spaces.Box(
                low=np.array(
                    [-5, -5, -np.pi, -5, -5, -5, -5, -10, -10, -np.pi, -np.pi, -10, -10, -10, -10, -10, -10, -np.pi,
                     -np.pi]),
                high=np.array([5, 5, np.pi, 5, 5, 5, 5, 10, 10, np.pi, np.pi, 10, 10, 10, 10, 10, 10, np.pi, np.pi]),
                dtype=np.float32))
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
        self.gazebo_reset = Gazebo_reset()
        # TODO:目标点设置
        self.choose_target = np.array([[2, 2], [0, 0]])
        self.target = random.choice(self.choose_target)
        # self.target = self.choose_target[1]
        self.is_success = [False] * self.agent_number
        self.gazebo_reset.reset_target(self.target)
        self.wall_collide = False
        self.agent_colliside = False

    def reset(self):
        """
        Reset environment
        Usage:
            obs = env.reset()
        """
        # rospy.logdebug("\nStart Environment Reset")
        # set init pose
        self.is_success = [False] * self.agent_number
        self.target = random.choice(self.choose_target)
        # self.target = self.choose_target[1]

        # print('target', self.target)
        # self.gazebo_reset.resetSim()
        # self.gazebo_reset.resetWorld()
        self.gazebo_reset.reset_agent_state(ddr_list=self.ddr_list)
        self.gazebo_reset.reset_target(self.target)
        self.wall_collide = False
        self.agent_colliside = False
        # IF RESET COLLIDE
        self._get_obs()
        if np.min(self.obs_dict[0]['delta_dist']) < 0.5 or np.min(self.obs_dict[1]['delta_dist']) < 0.5 or np.min(
                self.obs_dict[2]['delta_dist']) < 0.5:
            # self.gazebo_reset.resetSim()
            # self.gazebo_reset.resetWorld()
            self.gazebo_reset.reset_agent_state(ddr_list=self.ddr_list)
            self.gazebo_reset.reset_target(self.target)
        # rospy.logerr("\nEnvironment Reset!!!\n")
        self._get_obs()
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
        count_total = 0
        self._take_action(action_n)
        for i in range(self.agent_number):
            # update status
            reward, done, count = self._compute_reward(i, self.target)
            count_total += count
            if i == 0 or 1:
                safe = self._compute_safe(self.obs[i], self.obs[2], action_n[i], action_n[2])
                safe_01 = self._compute_safe(self.obs[0], self.obs[1], action_n[0], action_n[1])
                safe_10 = self._compute_safe(self.obs[1], self.obs[0], action_n[1], action_n[0])
                if i == 0:
                    safe += safe_01
                elif i == 1:
                    safe += safe_10
                safe_n.append(safe)
            # self.prev_obs = self.obs.copy() # make sure this happened after reward computing
            done_n.append(done)
            reward_n.append(reward)
        reward_n[2] = 0
        # safe_n.append(0)
        # print(safe_n)
        # safe_n = np.array(safe_n).flatten()
        # print(self.count_reach)
        if all(self.is_success):
            done_n = [True] * self.agent_number
            print('done!')
        # print(done_n)
        info = self.status
        self._get_obs()

        # rospy.logdebug("\nEnd environment step\n")
        # print(reward)
        # print(done_n)
        return self.obs, reward_n, done_n, safe_n, count, info

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
            [model_position, model_attitude, _, _, _] = self.gazebo_reset.get_state(ddr_name)
            positions_all.append(model_position)
            attitudes_all.append(model_attitude[2])  # 只需要欧拉角z
        positions_all = np.array(positions_all)
        attitudes_all = np.array(attitudes_all)
        positions_all = np.delete(positions_all, 2, axis=1)  # 去掉z轴位置
        for i, position in enumerate(positions_all):
            # 对于一个agent来说
            delta_list = positions_all - position
            delta_list = np.delete(delta_list, i, axis=0)
            # print(delta_list)
            dist_list = []
            theta_list = []
            for delta in delta_list:
                dist = np.linalg.norm(delta)
                if delta[0] != 0:
                    theta = np.arctan(delta[1] / delta[0])
                else:
                    theta = 0

                dist_list.append(dist)
                theta_list.append(theta)
            position = np.array(position).flatten()
            delta_pos = np.array(delta_list).flatten()
            delta_dist = np.array(dist_list).flatten()
            theta_array = np.array(theta_list).flatten()
            barrier_pos, wall_pos = self.__pos_wall_barrier(position)
            obs_one_agent_dict = {'position': position, 'euler': [attitudes_all[i]], 'delta_pos': delta_pos,
                                  'delta_dist': delta_dist, 'theta_array': theta_array, 'target': self.target}
            attitudes_rest = list(copy.deepcopy(attitudes_all))
            # print(attitudes_rest)
            del attitudes_rest[i]
            obs_one_agent = np.concatenate((position, [attitudes_all[i]], delta_pos, delta_dist, theta_array, np.array(
                self.target) - np.array(position), barrier_pos, wall_pos, attitudes_rest))
            # print(np.shape(obs_one_agent))
            # obs_one_agent = np.concatenate((position, [attitudes_all[i]], self.target))
            obs_all.append(obs_one_agent)
            obs_dict.append(obs_one_agent_dict)
        # obs共19维：包括自己的位置（2）(x,y)、，自身欧拉角（1）(euler), 与其他agent的相对位置（4）(dx1,dy1,dx2,dy2)，
        # 与其他agent的相对距离（2）(dist1,dist2)，相对位置方向夹角（2）(theta1,theta2),与目标点的相对位置（2）（dx,dy）,
        # 最近的障碍物位置（2）（x,y）,最近的墙位置（2）(x,y),其他两个的欧拉角(2)（euler1,euler2）
        self.obs = copy.deepcopy(obs_all)
        self.obs_dict = copy.deepcopy(obs_dict)
        return self.obs

    def __pos_wall_barrier(self, position):
        barrier_positions = []
        dist_agent2barriers = []
        for barrier in self.barrier_name:
            [barrier_position, _, _, _,
             _] = self.gazebo_reset.get_state(barrier)
            barrier_positions.append(barrier_position)
            dist_agent2barrier = np.linalg.norm(
                np.array(position) - np.array(barrier_position[0:2]))
            dist_agent2barriers.append(dist_agent2barrier)
        barrier_pos = barrier_positions[np.argmin(dist_agent2barriers)]

        dist_agent2walls = []
        wall_positions = []
        for wall in self.walls_name:
            [wall_position, _, _, _, _] = self.gazebo_reset.get_state(wall)
            if wall == 'wall_0' or wall == 'wall_1':
                self.dist_agent2wall = np.abs(
                    position[1] - wall_position[1])
                wall_position[0] = position[0]
            else:
                self.dist_agent2wall = np.abs(
                    position[0] - wall_position[0])
                wall_position[1] = position[1]

            dist_agent2walls.append(self.dist_agent2wall)
            wall_positions.append(wall_position)

        wall_pos = wall_positions[np.argmin(dist_agent2walls)]
        # print('agent_index', agent_index)
        # print(dist_agent2walls)
        # print(wall_positions)
        # print(wall_pos)
        # print('================')
        return barrier_pos[0:2], wall_pos[0:2]

    def __dist2wall_barrier(self, agent_index):
        barrier_positions = []
        self.dist_agent2barriers = []
        for barrier in self.barrier_name:
            [barrier_position, _, _, _, _] = self.gazebo_reset.get_state(barrier)
            barrier_positions.append(barrier_position)
            dist_agent2barrier = np.linalg.norm(
                np.array(self.obs_dict[agent_index]['position']) - np.array(barrier_position[0:2]))
            self.dist_agent2barriers.append(dist_agent2barrier)
        barrier_pos = barrier_positions[np.argmin(self.dist_agent2barriers)]

        self.dist_agent2walls = []
        wall_positions = []
        for wall in self.walls_name:
            [wall_position, _, _, _, _] = self.gazebo_reset.get_state(wall)
            if wall == 'wall_0' or wall == 'wall_1':
                self.dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][1] - wall_position[1])
                wall_position[0] = self.obs_dict[agent_index]['position'][0]
            else:
                self.dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][0] - wall_position[0])
                wall_position[1] = self.obs_dict[agent_index]['position'][1]

            self.dist_agent2walls.append(self.dist_agent2wall)
            wall_positions.append(wall_position)

        wall_pos = wall_positions[np.argmin(self.dist_agent2walls)]
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
        # self.gazebo_reset.unpausePhysics()
        barrier_pos0, wall_pos0 = self.__dist2wall_barrier(agent_index=0)
        barrier_pos1, wall_pos1 = self.__dist2wall_barrier(agent_index=1)

        state1 = np.concatenate((self.obs[2][0:3], self.obs[0][0:3], actions[2],
                                 self.obs[1][0:3], actions[1], barrier_pos0[0:2], wall_pos0[0:2]))
        state2 = np.concatenate((self.obs[2][0:3], self.obs[1][0:3], actions[2],
                                 self.obs[0][0:3], actions[0], barrier_pos1[0:2], wall_pos1[0:2]))
        state3 = np.concatenate((np.array([10, 10, 10]), self.obs[2][0:3], [
            0], np.array([10, 10, 10]), [0], barrier_pos1[0:2], wall_pos1[0:2]))  ##追击者的CBF
        # print(state2)
        # actions[0] = CBF(actions[0], state1)
        # actions[1] = CBF(actions[1], state2)
        # actions[2] = CBF(-actions[2], state3)
        vel_list = []
        for i, action in enumerate(actions):
            # 角速度为负逆时针转动，为正顺时针转动
            vel = Twist()
            vel.linear.x = self.v_linear
            if i == 2:
                vel.linear.x = self.v_linear * 0.8
            vel.angular.z = action
            vel_list.append(vel)
        # print('********************')
        # print('as', vel_list)
        # print('====================')
        # print('#####################')
        # print(actions)
        # print('###################')
        '''==============================='''
        for _ in range(1):
            # for i, vel in enumerate(vel_list):
            #     self.ddr_list[i].car_vel.publish(vel)

            self.ddr_list[0].car_vel.publish(vel_list[0])
            self.ddr_list[1].car_vel.publish(vel_list[1])
            self.ddr_list[2].car_vel.publish(vel_list[2])
            self.rate.sleep()
        '''=================================='''
        # self.gazebo_reset.pausePhysics()
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
        count = 0
        dist_agent2target = np.linalg.norm(
            np.array(self.obs_dict[agent_index]['position']) - np.array(target))
        reward, done = 0, False
        if agent_index == 0 or agent_index == 1:
            if dist_agent2target < 0.25:
                # reward += 1000  # before 2021/11/24/11:53
                reward += 1000
                if agent_index == 0 or agent_index == 1:
                    self.is_success[agent_index] = True
                    # self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                done = True

            # reward += -2.0 * dist_agent2target  # before 2021/11/24/11:53
            reward += -0.1 * dist_agent2target
            # if dist_agent2target < 1.415:
            #     reward += 1 / dist_agent2target 

        # for pos in self.obs_dict[agent_index]['position']:
        #     if np.abs(pos)>3:
        #         # self.gazebo_reset.reset_agent_state(self.ddr_list)
        #         reward -= 500
        #         done = True

        # 各个智能体间的距离
        # for d in self.obs_dict[agent_index]['delta_dist']:
        #     if d < 0.51:
        #         reward -= 1000
        #         # done = True
        #         self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])

        #     else:
        #         reward += 0

        for i, d in enumerate(self.obs_dict[agent_index]['delta_dist']):
            if d < 0.25:
                reward -= 1000
                # reward -= 100  # before 2021/11/24/11:53
                done = True
                if agent_index == 0 or agent_index == 1:
                    count = count + 1
                if agent_index == 2 and (
                        np.min(self.dist_agent2walls) <= 0.5 or np.min(self.dist_agent2barriers) <= 0.5):
                    self.wall_collide = True
                if agent_index == 2 and np.min(self.dist_agent2walls) > 0.5 and np.min(self.dist_agent2barriers) > 0.5:
                    self.agent_colliside = True
                # if agent_index == 0 or agent_index == 1:
                #     self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])

            # elif agent_index == 0 or 1 and i != 2:     # modify :if good agents between bad agent, reward +=d
            #     reward += 0.05 * d
            # elif agent_index == 1 and i == 2:     # modify :if good agents between bad agent, reward +=d
            #     reward += -0.5*1/d
            # elif agent_index == 0 and i == 2:
            #     reward += -0.5*1/d
            # else:
            #     reward += 0.2*d
        # rospy.logdebug("\nEnd Computing Reward\n")

        # 碰到障碍物奖励
        for barrier in self.barrier_name:
            [barrier_position, _, _, _, _] = self.gazebo_reset.get_state(barrier)
            dist_agent2barrier = np.linalg.norm(
                np.array(self.obs_dict[agent_index]['position']) - np.array(barrier_position[0:2]))
            if dist_agent2barrier < 0.25:
                reward -= 1000
                # self.wall_collide = True
                # reward -= 100  # before 2021/11/24/11:53
                # self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                if agent_index == 0 or agent_index == 1:
                    count = count + 1
                done = True

        # 碰到墙奖励  # after 2021/11/25/9:33, been used
        for wall in self.walls_name:
            [wall_position, _, _, _, _] = self.gazebo_reset.get_state(wall)
            if wall == 'wall_0' or wall == 'wall_1':
                self.dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][1] - wall_position[1])
            else:
                self.dist_agent2wall = np.abs(
                    self.obs_dict[agent_index]['position'][0] - wall_position[0])
            if agent_index == 0 or agent_index == 1 or agent_index == 2:

                if self.dist_agent2wall < 0.25:
                    reward -= 1000
                    if agent_index == 0 or agent_index == 1:
                        count = count + 1
                    # self.gazebo_reset.reset_agent_state([self.ddr_list[agent_index]])
                    done = True
                    self.wall_collide = True

                # else:  # add reward of wall
                #     reward += 0
                #     if agent_index == 1 or agent_index == 0:
                #         # reward += -0.4 * 1/self.dist_agent2wall  # before 2021/11/22/11:53
                #         reward += -0.1 * 1/self.dist_agent2wall

        return reward, done, count

    def _safe_constrains(self, good_obs, bad_obs, good_action, bad_action):
        '''
        good_obs: good agent's state
        bad_obs: bad agent's state
        good_action: good agent's action
        bad_action: bad agent's action
        '''
        Vp = copy.deepcopy(self.v_linear)
        Ve = copy.deepcopy(self.v_linear)
        r = self.safe_radius
        Xe, Ye, th_e = good_obs[0], good_obs[1], good_obs[2]
        Xp, Yp, th_p = bad_obs[0], bad_obs[1], bad_obs[2]
        sinp, cosp, sine, cose = np.sin(th_p), np.cos(th_p), np.sin(th_e), np.cos(th_e)
        Dp = np.array([Xp - Xe, Yp - Ye])
        Dv = np.array([Vp * cosp - Ve * cose, Vp * sinp - Ve * sine])
        Dot_vp = np.array([-Vp * sinp, Vp * cosp])
        Dot_ve = np.array([-Ve * sine, Ve * cose])

        # TODO: 根据h的实时情况调整k0,k1,
        p1, p2 = 2, 2
        K1 = p1 + p2
        K0 = p1 * p2

        # _safe_cons = 2 * (Vp ** 2 + Ve ** 2 - 2 * Vp * Ve * np.cos(th_p - th_e)) + \
        #     2 * bad_action * np.einsum('i,i->', Dp, Dot_vp) -\
        #     2 * good_action * np.einsum('i,i->', Dp, Dot_ve) + \
        #     2 * K1 * (np.einsum('i,i->', Dp, Dv)) +\
        #     K0 * (np.einsum('i,i->', Dp, Dp) - r ** 2)
        _safe_cons = np.sum(np.square(Dp)) - r ** 2
        return _safe_cons

    def _compute_safe(self, good_obs, bad_obs, good_action, bad_action):
        """
        Compute safe value based on current status for all good agents
        Return:
            safe_value  效果类比于 reward
        """
        h = self._safe_constrains(good_obs, bad_obs, good_action, bad_action)  # CBF
        # TODO: safe value的表达需要再调整
        safe_value = -np.exp(-h)
        # print('###############0')
        # print(safe_value)
        # print('###############1')
        # safe_value = 10 * h
        if h < 0:
            safe_value = -200

        return safe_value

    def rewardshaping(self, state, next_state, target_state, df=0.8):

        # computes the expert shapes for the given state -> state1
        # transitions, weighted by the posterior weights
        # weights = self.weights
        # experts = self.experts
        shape, next_shape = 0.0, 0.0
        shape = []
        next_shape = []
        # reward shaping:
        # norm_target_State = self.s_norm(self.observation_space, target_state)
        # norm_State = self.s_norm(self.observation_space, state)
        # norm_next_State = self.s_norm(self.observation_space, next_state)
        # shape = - np.sqrt(np.square(
        #     state[11] - target_state[0]) + np.square(state[12] - target_state[1]))
        # next_shape = - np.sqrt(np.square(next_state[11] - target_state[0]) + np.square(
        #     next_state[12] - target_state[1]))
        for i in range(self.agent_number):
            # print(state)
            # print(state[i][11])
            shape.append(np.sqrt(np.square(state[i][11] - target_state[0]) + np.square(state[i][12] - target_state[1])))

            next_shape.append(np.sqrt(
                np.square(next_state[i][11] - target_state[0]) + np.square(next_state[i][12] - target_state[1])))

        return df * np.array(next_shape) - np.array(shape)


if __name__ == "__main__":
    env = TAI_Env()
    num_steps = int(1e5)
    # obs = env.reset()
    ep, st = 0, 0
    o = env.reset()
    import sys

    sys.path.append("..")
    from controllers.simple_reach import simple_reach
    from controllers.simple_pv import simple_pursuit, simple_avoid

    for t in range(num_steps):
        # 使用随机action测试
        # actions = random.random(3) - 0.5
        # print('sd', actions)
        '''===========使用简单的到达目标点测试==========='''
        actions = []
        for i, ob in enumerate(o):
            a = simple_reach(ob, env.target)
            actions.append(a)
            """==========================================="""
            """当episode大于某个数时，随机一个智能体变为追击离它最近的智能体，相应地智能体也要变为躲避它"""
        if ep >= 1:
            print('start')
            if ep == 1:
                bad_index = random.choice(range(env.agent_number))
                dist = [np.linalg.norm(o[i][0:2] - o[bad_index][0:2]) for i in range(env.agent_number)]
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
