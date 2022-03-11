#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========safe-rl-cbf===============
@File: rl_ros_robotarium_env
@Time: 2022/3/11 下午8:09
@Author: chh3213
@Description: rl-cbf layer算法使用的ros环境
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
from robotarium_get import Robotarium
from control_car import DDR
import random


class RobotariumEnv(core.Env):
    def __init__(self):
        self.name = 'safe_robotarium_attribute'
        rospy.init_node(self.name, anonymous=True)
        # Define action and observation space
        # They must be gym.spaces objects
        # TODO：代码运行频率星需要根据实际情况调整
        self.rate = rospy.Rate(20)
        self.agent_number = 3
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(7,))
        self.ddr_list = []
        self.ddr_name = []
        for i in range(self.agent_number):
            ddr = DDR('ddr_' + str(i))
            self.ddr_list.append(ddr)
            self.ddr_name.append('ddr_' + str(i))

        # Initialize Env
        self.episode_step = 0
        # TODO:目标点设置、障碍物设置
        self.goal_size = 0.3  # 目标点大小
        self.goal_pos = np.array([1.5, 1.5])
        self.dt = 0.02
        self.robotarium_state = Robotarium()
        self.robotarium_state.reset_target(self.goal_pos)
        # 障碍物位置
        # ros
        self.barrier_name = ['barrier_0', 'barrier_1', 'barrier_2']
        self.hazards_locations = np.array([[-1., 0.], [1., 0], [0, 1.]])
        self.hazards_radius = 0.3  # 障碍物半径
        self.robotarium_state.reset_barrier(self.hazards_locations)
        self.max_episode_steps = 400

    def step(self, action):
        """Organize the observation to understand what's going on

        Parameters
        ----------
        action : ndarray
                Action that the agent takes in the environment

        Returns
        -------
        new_obs : ndarray
          The new observation with the following structure:
          [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, dist2goal]

        """

        action = np.clip(action, -1.0, 1.0)
        states, reward, done, info = self._step(action)
        other_agent_s = np.delete(states[1:], 2, 1)  # 除去欧拉角
        other_s = np.vstack((other_agent_s, self.hazards_locations))
        return self.get_obs(), other_s, reward, done, info

    def _step(self, action):
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
        # 自己的agent
        vel_list = []
        vel = Twist()
        vel.linear.x = action[0]
        vel.angular.z = action[1]
        vel_list.append(vel)
        # 其他agent--任意给定
        for i in range(1, self.agent_number):
            vel = Twist()
            vel.linear.x = 0.25
            vel.angular.z = np.random.rand() * 0.5 - 0.25
            vel_list.append(vel)

        for i, vel in enumerate(vel_list):
            # print(vel)
            self.ddr_list[i].car_vel.publish(vel)
        # self.rate.sleep()

        reward, done, info = self._reward_done()
        self.states = self.get_pose()
        self.episode_step += 1

        return self.states, reward, done, info

    def _reward_done(self):
        info = dict()
        reward = 0
        done = False
        dist_goal = self._goal_dist()

        self_state = self.states[0]
        other_states = self.states[1:]

        # # Check boundaries
        # if(self_state[1]>1.9 or self_state[1]<-1.9 or self_state[0]>1.9 or self_state[0]<-1.9):
        #     print('Out of boundaries !!')
        #     reward -= 100
        #     done =True

        for idx in range(np.size(other_states, 0)):
            distSqr = (self_state[0] - other_states[idx][0]) ** 2 + (self_state[1] - other_states[idx][1]) ** 2
            if distSqr < (0.25) ** 2:
                print('Get caught, mission failed !')
                done = True
                reward -= 100

        for idx in range(len(self.hazards_locations)):
            distSqr = (self_state[0] - self.hazards_locations[idx][0]) ** 2 + (
                    self_state[1] - self.hazards_locations[idx][1]) ** 2
            if distSqr < (0.25) ** 2:
                print('hit barrier!')
                done = True
                reward -= 500
        # Check if goal is met
        if self.goal_met():
            print('Reach goal successfully!')
            info['goal_met'] = True
            reward += 500
            done = True
        else:
            reward -= 0.1 * dist_goal
            # reward += 10.0 * (self.last_goal_dist - dist_goal)

        self.last_goal_dist = dist_goal

        if self.max_episode_steps <= self.episode_step:
            done = True
        # Include constraint cost in reward
        if np.any(np.sum((self_state[:2] - self.hazards_locations) ** 2, axis=1) < self.hazards_radius ** 2):
            if 'cost' in info:
                info['cost'] += 0.1
            else:
                info['cost'] = 0.1

        return reward, done, info

    def goal_met(self):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """
        self_state = self.states[0]
        return np.linalg.norm(self_state[:2] - self.goal_pos) <= self.goal_size

    def reset(self, is_show_figure=False):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """
        self.robotarium_state.resetWorld()
        self.robotarium_state.reset_agent_state(ddr_list=self.ddr_list)
        self.robotarium_state.reset_target(self.goal_pos)

        self.states = self.get_pose()

        self.episode_step = 0
        other_agent_s = np.delete(self.states[1:], 2, 1)  # 除去欧拉角
        other_s = np.vstack((other_agent_s, self.hazards_locations))

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()

        return self.get_obs(), other_s

    def close(self):
        pass

    def get_obs(self):
        """Given the state, this function returns it to an observation akin to the one obtained by calling env.step

        Parameters
        ----------

        Returns
        -------
        observation : ndarray
          Observation: [pos_x, pos_y, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
        """
        self_state = self.states[0]
        rel_loc = self.goal_pos - self_state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass()  # compass to the goal

        return np.array([self_state[0], self_state[1], np.cos(self_state[2]), np.sin(self_state[2]), goal_compass[0],
                         goal_compass[1], np.exp(-goal_dist)])

    def obs_compass(self):
        """
        Return a robot-centric compass observation of a list of positions.
        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.
        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

        # Get ego vector in world frame
        self_state = self.states[0]
        vec = self.goal_pos - self_state[:2]
        # Rotate into frame
        R = np.array([[np.cos(self_state[2]), -np.sin(self_state[2])], [np.sin(self_state[2]), np.cos(self_state[2])]])
        vec = np.matmul(vec, R)
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec

    def _goal_dist(self):
        self_state = self.states[0]
        return np.linalg.norm(self.goal_pos - self_state[:2])

    def get_pose(self):
        """Returns the states of the agents.

        -> 3xN numpy array (of robot poses)
        """
        poses = np.empty(3)
        for ddr in self.ddr_name:
            [model_position, model_attitude, _, _, _] = self.robotarium_state.get_state(ddr)
            pose = np.hstack((np.array(model_position[0:2]), np.array(model_attitude[2])))
            temp = copy.deepcopy(pose)
            poses = np.vstack((poses, temp))
        poses = np.delete(poses, 0, 0)  # Nx3
        poses = poses.T  # 3xN
        return poses
