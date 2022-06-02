import time

import numpy as np
import random
import gym
from gym import spaces, core
from rps.robotarium import Robotarium
from matplotlib import patches
import matplotlib.pyplot as plt
import rospy


class RobotariumEnv(core.Env):
    def __init__(self, args=None):
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(7,))
        # self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(11,))
        # 障碍物位置
        # self.hazards_locations = np.array([[-1.5, 0., 0.], [1.5, 0., 0.], [0.0, 1.5, 0.], [0.0, -1.5, 0.]])
        # self.hazards_locations = np.array([[1.5, 0, 0]])
        self.hazards_locations = np.array([[1, 1., 0.], [1.5, 0., 0.], [0.0, 1.5, 0.]])
        self.hazards_radius = 0.25  # 障碍物半径
        self.args = args
        if args.mode != "train":
            self.max_episode_steps = 3000  # eval
            # self.agent_number = 10
            self.agent_number = 3
        else:
            self.max_episode_steps = 3000
            self.agent_number = 2
        self.goal_size = 0.3  # 目标点大小
        # Initialize Env
        self.episode_step = 0
        self.collison_obstacle = 0
        self.get_caught = 0
        self.dt = 0.02
        # TODO goal pose 写成所有agent的目标
        self.goal_pos = [np.array([4, 4]) for _ in range(self.agent_number)]
        # self.goal_train_random = np.array([[0, 0], [3, 3], [-3, 3], [-3, -3], [3, -3]]) #原训练
        self.goal_train_random = np.array([[0, 0], [4, 4], [-4, 4], [-4, -4], [4, -4]]) #20220417 
        self.second = [False for _ in range(self.agent_number)]
        self.first = [True for _ in range(self.agent_number)]
        # agent 初始位置
        # a = np.array([[3. * np.random.rand() - 1.5, 3. * np.random.rand() - 1.5, 0]]).T
        if args.mode == "train":
            a = np.array([[6. * np.random.rand() - 3, 6. * np.random.rand() - 3, 6.28 * np.random.rand() - 3.14]]).T
            # a = np.array([[10. * np.random.rand() - 5, 10. * np.random.rand() - 5, 6.28 * np.random.rand() - 3.14]]).T
            self.initial_conditions = a
            for _ in range(1, self.agent_number):
                self.initial_conditions = np.concatenate((self.initial_conditions, np.array(
                    [[6 * np.random.rand() - 3.0, 6 * np.random.rand() - 3.0, 6.28 * np.random.rand() - 3.14]]).T),
                                                         axis=1)
        else:
            # a = np.array([[0., 0., 0]]).T
            # self.initial_conditions = a
            # for i in range(1, self.agent_number):
            #     self.initial_conditions = np.concatenate((self.initial_conditions, np.array(
            #         [[- i % 3 + 2, (i / 3 + 1) * (-1), 0]]).T), axis=1)
            if self.agent_number == 10:
                self.initial_conditions = np.array(
                    [[0., 0., 0], [-2, -1.5, 0], [0, -1.5, 0], [2, -1.5, 0], [-2, -3, 0], [0, -3, 0],
                     [2, -3, 0],
                     [-2, -4.5, 0],
                     [0, -4.5, 0], [2, -4.5, 0]]).T
            elif self.agent_number == 3:
                self.initial_conditions = np.array([[0, 0.5, 0], [0, 1, 0], [0, -1, 0]]).T

        self.agents = Robotarium(number_of_robots=self.agent_number, show_figure=False,
                                 initial_conditions=self.initial_conditions, sim_in_real_time=False)
        self.states = self.agents.get_poses().T
        self.agents.step()

    def step(self, actions):
        """
        Parameters
        ----------
        actions: agents' actons: each agent:[v,w]

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
        # print(np.shape(actions))
        # print(actions)
        u = actions
        start = time.time()
        self.agents.set_velocities(np.arange(self.agent_number), u.T)
        self.states = self.agents.get_poses().T  # shape[agent_numbe,2]，第一维为RL agent，其他为普通agent
        self.agents.step()

        self.episode_step += 1
        return self.states

    def _reward_done(self, index):
        info = dict()
        reward = 0
        done = False
        flag = False
        self.is_success = False
        dist_goal = self._goal_dist(index)

        self_state = self.states[index]
        other_agent_s = np.delete(self.states, index, 0)  # 除去自身
        # other_agent_s = np.delete(other_agent_s, 2, 1)  # 除去欧拉角
        # other_s = np.vstack((other_agent_s, self.hazards_locations))
        # # Check boundaries
        # if(self_state[1]>1.9 or self_state[1]<-1.9 or self_state[0]>1.9 or self_state[0]<-1.9):
        #     print('Out of boundaries !!')
        #     reward -= 100
        #     done =True

        # other_s = np.vstack((other_agent_s, self.hazards_locations))

        # """agent-nearest agent reward """
        # dist_list = []
        # # for o_s in self.hazards_locations: #4
        # # for o_s in other_s:
        # for o_s in other_agent_s:
        #     agent_dist = np.linalg.norm(o_s[0:2] - self_state[:2])
        #     dist_list.append(agent_dist)
        # ind = np.argmin(dist_list)
        # # reward += 0.002 * dist_list[ind]  # 050701
        # # reward += 0.005 * dist_list[ind]  # 050702
        # reward += 0.001 * dist_list[ind]  # 050801
        # # reward += 0.02 * dist_list[ind]  # 2
        # # reward += 0.03 * dist_list[index]  # 1

        for idx in range(np.size(other_agent_s, 0)):
            distSqr = (self_state[0] - other_agent_s[idx][0]) ** 2 + (self_state[1] - other_agent_s[idx][1]) ** 2
            if distSqr < self.hazards_radius ** 2:
                print('Get caught, mission failed !')
                self.get_caught +=1
                done = True
                # reward -= 1000
                reward = -1000
                return reward, done, info
        for idx in range(np.size(self.hazards_locations, 0)):
            distSqr = (self_state[0] - self.hazards_locations[idx][0]) ** 2 + \
                (self_state[1] - self.hazards_locations[idx][1]) ** 2
            if distSqr < self.hazards_radius ** 2:
                print('Get collision, mission failed !')
                done = True
                self.collison_obstacle+=1
                # reward -= 1000
                reward = -1000
                return reward, done, info
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
                self.goal_pos[index] = np.array([0, 0])
            elif self.goal_met(index) and self.second[index]:
                # print(self.goal_pos)
                print('Reach second goal successfully!')
                info['goal_met'] = True
                reward += 1000
                # done = True
                self.second[index] = False
                self.first[index] = False
                self.goal_pos[index] = np.array([2, 2])
            elif self.goal_met(index) and (self.goal_pos[index] == np.array([2, 2])).all():
                print('Reach first goal successfully again!')
                info['goal_met'] = True
                reward += 1000
                self.goal_pos[index] = np.array([0, 0])
            elif self.goal_met(index) and (self.goal_pos[index] == np.array([0, 0])).all():
                print('Reach first goal successfully again!')
                info['goal_met'] = True
                reward += 1000
                self.goal_pos[index] = np.array([2, 2])

            if not self.second[index] and not self.first[index]:
                self.is_success = True
                # done = True
        else:
            if self.goal_met(index):
                print('Reach goal successfully!')
                info['goal_met'] = True
                # reward += 1000
                reward = +1000
                done = True
            else:
                # reward -= 0.1 * dist_goal
                reward += 10 * (self.last_goal_dist - dist_goal)

        self.last_goal_dist = dist_goal

        if self.max_episode_steps <= self.episode_step:
            done = True
        # # Include constraint cost in reward
        # if np.any(np.sum((self_state[:2] - self.hazards_locations[:, :2]) ** 2, axis=1) < self.hazards_radius ** 2):
        #     if 'cost' in info:
        #         info['cost'] += 0.1
        #     else:
        #         info['cost'] = 0.1
        # print(reward)
        return reward, done, info
    
    def _is_success(self):
        """return whether succeed to real goal """
        return self.is_success

    def goal_met(self, index):
        """Return true if the current goal is met this step

        Returns
        -------
        goal_met : bool
            True if the goal condition is met.

        """
        self_state = self.states[index]
        return np.linalg.norm(self_state[:2] - self.goal_pos[index]) <= self.goal_size + 0.05

    def reset(self, is_show_figure=False):
        """ Reset the state of the environment to an initial state.

        Returns
        -------
        observation : ndarray
            Next observation.
        """
        plt.close()
        self.second = [False for _ in range(self.agent_number)]
        self.first = [True for _ in range(self.agent_number)]

        if self.args.mode == "train":
            self.goal_pos = [random.choice(self.goal_train_random) for _ in range(self.agent_number)]
            # a = np.array([[-1.5, -1.5 ,0]]).T
            a = np.array([[6. * np.random.rand() - 3, 6. * np.random.rand() - 3, 6.28 * np.random.rand() - 3.14]]).T  # z之前训练设置
            # a = np.array([[10. * np.random.rand() - 5, 10. * np.random.rand() - 5, 6.28 * np.random.rand() - 3.14]]).T
            self.initial_conditions = a
            for _ in range(1, self.agent_number):
                self.initial_conditions = np.concatenate((self.initial_conditions, np.array(
                    [[10. * np.random.rand() - 5, 10. * np.random.rand() - 5, 6.28 * np.random.rand() - 3.14]]).T),
                                                         axis=1)
        else:
            self.goal_pos = [np.array([2, 2]) for _ in range(self.agent_number)]
            # a = np.array([[0., 0., 0]]).T
            # self.initial_conditions = a
            # for i in range(1, self.agent_number):
            #     self.initial_conditions = np.concatenate((self.initial_conditions, np.array(
            #         [[- i % 3 - 1, (i / 3 + 1) * (-1), 0]]).T), axis=1)
            if self.agent_number == 10:
                self.initial_conditions = np.array(
                    [[0., 0., 0], [-2, -1.5, 0], [0, -1.5, 0], [2, -1.5, 0], [-2, -3, 0], [0, -3, 0],
                     [2, -3, 0],
                     [-2, -4.5, 0],
                     [0, -4.5, 0], [2, -4.5, 0]]).T
            elif self.agent_number == 3:
                self.initial_conditions = np.array([[0, 0.5, 0], [0, 1, 0], [0, -1, 0]]).T

        self.agents = Robotarium(number_of_robots=self.agent_number, show_figure=is_show_figure,
                                 initial_conditions=self.initial_conditions, sim_in_real_time=False)
        # if (is_show_figure):
        #     self.agents.axes.add_patch(
        #         patches.Circle(self.hazards_locations[0][0:2], self.hazards_radius - 0.1, fill=True))  # Obstacle 1
        #     self.agents.axes.add_patch(
        #         patches.Circle(self.hazards_locations[1][0:2], self.hazards_radius - 0.1, fill=True))  # Obstacle 2
        #     self.agents.axes.add_patch(
        #         patches.Circle(self.hazards_locations[2][0:2], self.hazards_radius - 0.1, fill=True))  # Obstacle
        #     self.agents.axes.add_patch(
        #         patches.Circle(self.hazards_locations[3][0:2], self.hazards_radius - 0.1, fill=True))  # Obstacle
        #     self.agents.axes.add_patch(
        #         patches.Circle(self.goal_pos[0], self.goal_size, fill=False, zorder=10))  # target
        #     self.agents.axes.add_patch(
        #         patches.Circle(np.array([0, 0]), self.goal_size, fill=False, zorder=10))  # target

        self.states = self.agents.get_poses().T
        self.agents.step()

        self.episode_step = 0
        # other_agent_s = np.delete(self.states, index, 0)
        # # other_agent_s = np.delete(other_agent_s, 2, 1)  # 除去欧拉角
        # other_s = np.vstack((other_agent_s, self.hazards_locations))

        # Re-initialize last goal dist
        # self.last_goal_dist = self._goal_dist(index)
        self.last_goal_dist = self._goal_dist(0)

        return self.states

    def render(self):
        pass

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
        other_agent_s = np.delete(self.states, index, 0)  # 除去自身
        other_s = np.vstack((other_agent_s, self.hazards_locations))
        other_s = np.delete(other_s, 2, 1)  # 除去欧拉角
        self_state = self.states[index]
        rel_loc = self.goal_pos[index] - self_state[:2]
        goal_dist = np.linalg.norm(rel_loc)
        goal_compass = self.obs_compass(index)  # compass to the goal

        return np.array([self_state[0], self_state[1], np.cos(self_state[2]), np.sin(self_state[2]), goal_compass[0],
                         goal_compass[1], np.exp(-goal_dist)]), other_s
        # return np.array([self_state[0], self_state[1], np.cos(self_state[2]), np.sin(self_state[2]), goal_compass[0],
        #                  goal_compass[1]])
        # return np.array([goal_compass[0],goal_compass[1], np.exp(-goal_dist)]), other_s

        """全部维度的训练，最近的智能体，全部障碍物，自己的相对位置"""
        # other_agent_s = np.delete(self.states, index, 0)  # 除去自身
        # other_s = np.vstack((other_agent_s, self.hazards_locations))
        # temp_obs = goal_compass[0:2]
        # temp_obs = np.concatenate((temp_obs, [np.exp(-goal_dist)]))  # 添加一维

        # dist_list = []
        # for o_s in other_agent_s:
        #     agent_dist = np.linalg.norm(o_s[0:2] - self_state[:2])
        #     dist_list.append(agent_dist)
        # idx = np.argmin(dist_list)
        # rel_o_s = other_agent_s[idx, 0:2] - self_state[:2]

        # temp_obs = np.concatenate((temp_obs, rel_o_s))

        # for o_s in self.hazards_locations:
        #     temp_obs = np.concatenate((temp_obs, o_s[0:2] - self_state[:2]))
        # return temp_obs

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
