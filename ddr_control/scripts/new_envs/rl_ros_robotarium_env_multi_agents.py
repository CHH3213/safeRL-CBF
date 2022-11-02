#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========safe-rl-cbf===============
@File: rl_ros_robotarium_env
@Time: 2022/6/10 下午8:09
@Author: chh3213
@Description: 作为10个agents的测试环境，3个agents仅仅目标点和障碍物的位置不同
10个agents：目标点为(1,1),(3,3),障碍物为(2,2)
3个agents：目标点为(0,0),(2,2),障碍物为(1,1)
========Above the sun, full of fire!=============
"""
from __future__ import absolute_import, division, print_function
import gym
from gym import spaces, core
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
from ros_robotarium_get import Robotarium
from control_robotarium_ros import DDR
import random


class RobotariumEnv(core.Env):
    def __init__(self, args):
        self.name = 'safe_robotarium_attribute'
        rospy.init_node(self.name, anonymous=True)
        # Define action and observation space
        # They must be gym.spaces objects
        # TODO：代码运行频率星需要根据实际情况调整
        self.rate = rospy.Rate(50)
        self.args = args

        if args.mode != "train":
            self.max_episode_steps = 5000  # eval
            self.agent_number = 10
        else:
            self.max_episode_steps = 1500
            self.agent_number = 1
        self.ddr_list = []
        self.ddr_name = []
        for i in range(self.agent_number):
            ddr = DDR('ddr_' + str(i))
            self.ddr_list.append(ddr)
            self.ddr_name.append('ddr_' + str(i))

        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(1,))
        # self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(3,))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(6,))

        # Initialize Env
        self.episode_step = 0
        self.collison_obstacle = 0
        self.get_caught = 0
        # TODO:目标点设置、障碍物设置
        self.goal_size = 0.25  # 目标点大小
        # TODO:每两辆小车两个目标点来回  0~8作为可控小车
        if args.agent_number==3:
            next_goal = 2
            origin_goal = 0
        else:
            next_goal = 3
            origin_goal = 1
        
        self.goal_pos_next = np.array([
                                np.array([next_goal, next_goal]), np.array([next_goal, next_goal]),
                                np.array([-next_goal, next_goal]), np.array([-next_goal, next_goal]),
                                np.array([-next_goal, -next_goal]), np.array([-next_goal, -next_goal]),
                                np.array([next_goal, -next_goal]), np.array([next_goal, -next_goal])
                                ])
        
        self.goal_pos_origin = np.array([
                                np.array([origin_goal, origin_goal]), np.array([origin_goal, origin_goal]),
                                np.array([-origin_goal, origin_goal]), np.array([-origin_goal, origin_goal]),
                                np.array([-origin_goal, -origin_goal]), np.array([-origin_goal, -origin_goal]),
                                np.array([origin_goal, -origin_goal]), np.array([origin_goal, -origin_goal])
                                ])
        

        self.goal_pos = copy.deepcopy(self.goal_pos_next)
        self.second = [False for _ in range(self.agent_number)]
        self.first = [True for _ in range(self.agent_number)]
        self.dt = 0.02
        self.robotarium_state = Robotarium()
        self.is_success_2 = False

        # self.robotarium_state.reset_target(self.goal_pos)
        # 障碍物位置
        # ros
        self.barrier_name = ['barrier_0', 'barrier_1', 'barrier_2']
        locations = 2
        if self.agent_number == 10:
            self.hazards_locations = np.array([[1.5, 0, 0.],[-1.5, 0, 0.], [0, -1.5, 0.], [0, 1.5, 0.],[locations, locations, 0.], [-locations, locations, 0.], [-locations, -locations, 0.], [locations, -locations, 0.]])
        elif self.agent_number==3:
            self.hazards_locations = np.array([[1.5, 0, 0.],[-1.5, 0, 0.],  [0, 1.5, 0.], [locations, locations, 0.]])
        self.hazards_radius = 0.25  # 障碍物半径
        # self.robotarium_state.reset_barrier(self.hazards_locations)

    def step(self, actions):
        """

        Parameters
        ----------
        action: agents' actons: each agent:[v,w]

        Returns
        -------
        state : ndarray
            New internal state of the agent.
        reward : float
            Reward collected during this transition.
        done : bool
            Whether the episode terminated.
        info : dict
            Additional info relevant to the environment.
        """
        s = time.time()
        vel_list = []
        for i in range(self.agent_number):
            vel = Twist()
            vel.linear.x = actions[i][0]
            # vel.angular.z = np.random.rand() * 0.5 - 0.25
            vel.angular.z = actions[i][1]
            vel_list.append(vel)

        for i, vel in enumerate(vel_list):
            # print(vel)
            self.ddr_list[i].car_vel.publish(vel)
        self.states = self.get_pose().T
        # print("waste_time", time.time() - s)
        self.episode_step += 1
        # self.rate.sleep()

        return self.states

    def _reward_done(self, index):
        # print('1',self.first)
        # print('2',self.second)
        info = dict()
        reward = 0
        done = False
        dist_goal = self._goal_dist(index)
        self.is_success = False

        self_state = self.states[index]
        other_agent_s = np.delete(self.states, index, 0)  # 除去自身
        other_states = np.vstack((other_agent_s, self.hazards_locations))

        # # Check boundaries
        # if(self_state[1]>1.9 or self_state[1]<-1.9 or self_state[0]>1.9 or self_state[0]<-1.9):
        #     print('Out of boundaries !!')
        #     reward -= 100
        #     done =True
        # TODO："""寻找最近的障碍物"""
        # hazard_dist_list = []
        # for o_s in self.hazards_locations:
        #     hazard_dist = np.linalg.norm(o_s[0:2] - self_state[:2])
        #     hazard_dist_list.append(hazard_dist)
        # ind = np.argmin(hazard_dist_list)
        # self.nearest_loca = copy.deepcopy(self.hazards_locations[ind])
        """======================================="""
        for idx in range(np.size(other_agent_s, 0)):
            distSqr = (self_state[0] - other_agent_s[idx][0]) ** 2 + \
                (self_state[1] - other_agent_s[idx][1]) ** 2
            if distSqr < (self.hazards_radius-0.1) ** 2:
                print('Get caught, mission failed !')
                self.get_caught += 1
                done = True
                reward -= 1000
        for idx in range(np.size(self.hazards_locations, 0)):
            distSqr = (self_state[0] - self.hazards_locations[idx][0]) ** 2 + \
                (self_state[1] - self.hazards_locations[idx][1]) ** 2
            if distSqr < (self.hazards_radius-0.1) ** 2:
                print('Get collision, mission failed !')
                done = True
                self.collison_obstacle += 1
                reward -= 1000
        # print(self.goal_met(index))
        # print(self.states[0])
        if self.args.mode != "train":
            """======goal reward==========="""
            # Check if goal is met
            if self.goal_met(index) and self.first[index]:
                print('Reach first goal successfully!')
                info['goal_met'] = True
                reward += 1000
                # done = True
                self.second[index] = True
                self.first[index] = False
                self.goal_pos[index] = self.goal_pos_origin[index]
            elif self.goal_met(index) and self.second[index]:
                # print(self.goal_pos)
                print('Reach second goal successfully!')
                info['goal_met'] = True
                reward += 1000
                # done = True
                self.second[index] = False
                self.first[index] = False
                self.goal_pos[index] = self.goal_pos_next[index]
            elif self.goal_met(index) and (self.goal_pos[index] == self.goal_pos_next[index]).all():
                print('Reach first goal successfully again!')
                info['goal_met'] = True
                reward += 1000
                self.goal_pos[index] = self.goal_pos_origin[index]
            elif self.goal_met(index) and (self.goal_pos[index] == self.goal_pos_origin[index]).all():
                print('Reach second goal successfully again!')
                info['goal_met'] = True
                reward += 1000
                self.is_success_2 = True
                # print(self.is_success_2)

                self.goal_pos[index] = self.goal_pos_next[index]
            if not self.second[index] and not self.first[index]:
                # done = True
                self.is_success = True

        else:
            if self.goal_met(index):
                print('Reach goal successfully!')
                info['goal_met'] = True
                reward += 1000
                done = True
            else:
                reward -= 0.1 * dist_goal
                # reward += 10 * (self.last_goal_dist - dist_goal)

        self.last_goal_dist = dist_goal

        if self.max_episode_steps <= self.episode_step:
            done = True
        # # Include constraint cost in reward
        # if np.any(np.sum((self_state[:2] - self.hazards_locations[:, :2]) ** 2, axis=1) < self.hazards_radius ** 2):
        #     if 'cost' in info:
        #         info['cost'] += 0.1
        #     else:
        #         info['cost'] = 0.1
        return reward, done, info

    def _is_success(self):
        """return whether succeed to real goal """
        return self.is_success, copy.deepcopy(self.is_success_2)
    
    def nearest_obstacle(self,index):
        """
        返回距离当前索引为index的智能体最近的障碍物的状态信息
        """
        hazard_dist_list = []
        self_state = self.states[index]
        for o_s in self.hazards_locations:
            hazard_dist = np.linalg.norm(o_s[0:2] - self_state[:2])
            hazard_dist_list.append(hazard_dist)
        ind = np.argmin(hazard_dist_list)
        self.nearest_loca = copy.deepcopy(self.hazards_locations[ind])
        return self.nearest_loca

    def nearest_agent(self, index):
        """
        返回当前索引为index的智能体和距离当前索引为index的智能体最近的智能体的状态信息
        """
        dist_list = []
        self_state = self.states[index]
        other_agent_s = np.delete(self.states, index, 0)  # 除去自身
        for o_s in other_agent_s:
            dist = np.linalg.norm(o_s[0:2] - self_state[:2])
            dist_list.append(dist)
        ind = np.argmin(dist_list)
        
        # return np.vstack((self_state, other_agent_s[ind]))
        ind2 = np.argsort(dist_list)[1]
        return np.vstack((other_agent_s[ind], other_agent_s[ind2]))

    def goal_met(self, index):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """
        self_state = self.states[index]
        return np.linalg.norm(self_state[:2] - self.goal_pos[index]) <= self.goal_size 



    def reset(self, is_show_figure=False):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """
        self.robotarium_state.resetWorld()
        # self.robotarium_state.reset_test_agent_random_state(self.ddr_list)  # 测试100回合使用
        self.robotarium_state.reset_test_10agents_state(self.ddr_list)  # 测试时使用
        # self.robotarium_state.reset_target(self.goal_pos_next)
        self.goal_pos = copy.deepcopy(self.goal_pos_next)
        self.second = [False for _ in range(self.agent_number)]
        self.first = [True for _ in range(self.agent_number)]


        self.is_success, self.is_success_2 = False,False
        self.states = self.get_pose().T

        self.episode_step = 0
        # other_agent_s = np.delete(self.states[1:], 2, 1)  # 除去欧拉角
        # other_s = np.vstack((other_agent_s, self.hazards_locations))
        #
        # # Re-initialize last goal dist
        # self.last_goal_dist = self._goal_dist()

        return self.states

    def close(self):
        pass

    def get_obs(self, index):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
        """
        # other_agent_s = np.delete(self.states, index, 0)  # 除去自身
        # other_s = np.vstack((other_agent_s, self.hazards_locations))
        other_s = np.vstack((self.nearest_agent(index), self.nearest_obstacle(index)))
        # other_s = np.delete(other_s, 2, 1)  # 除去欧拉角
        self_state = self.states[index]
        rel_loc = self.goal_pos[index] - self_state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass(index)  # compass to the goal

        # return np.array([self_state[0], self_state[1], np.cos(self_state[2]), np.sin(self_state[2]), goal_compass[0],
        #                  goal_compass[1], np.exp(-goal_dist)]), other_s
        # dim:6
        return np.array([self_state[0], self_state[1], self_state[2], goal_compass[0],
                         goal_compass[1], np.exp(-goal_dist)]), other_s
    def obs_compass(self, index):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        self_state = self.states[index]
        vec = self.goal_pos[index] - self_state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self_state[2]), -np.sin(self_state[2])], [np.sin(self_state[2]), np.cos(self_state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist(self, index):
        self_state = self.states[index]
        return np.linalg.norm(self.goal_pos[index] - self_state[:2])

    def get_pose(self):
        """Returns the states of the agents.

        -> 3xN numpy array (of robot poses)
        """
        poses = np.empty(3)
        for ddr in self.ddr_name:
            # s = time.time()
            [model_position, model_attitude, _, _, _] = self.robotarium_state.get_state(ddr)
            # print("get state time ", time.time() - s)
            pose = np.hstack((np.array(model_position[0:2]), np.array(model_attitude[2])))
            temp = copy.deepcopy(pose)
            poses = np.vstack((poses, temp))
        poses = np.delete(poses, 0, 0)  # Nx3
        poses = poses.T  # 3xN
        # print('pose', poses)
        return poses
