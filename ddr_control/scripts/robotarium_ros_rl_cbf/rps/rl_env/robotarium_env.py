import numpy as np
import gym
from gym import spaces, core
from rps.robotarium import Robotarium
from matplotlib import patches
import matplotlib.pyplot as plt


class RobotariumEnv(core.Env):
    def __init__(self):
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.safe_action_space = spaces.Box(low=-2.5, high=2.5, shape=(2,))
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(7,))
        self.bds = np.array([[-3., -3.], [3., 3.]])
        # 障碍物位置
        self.hazards_locations = np.array([[0., 0.], [-1., 1.], [-1., -1.]])
        self.hazards_radius = 0.2  # 障碍物半径
        self.max_episode_steps = 400
        self.goal_size = 0.3  # 目标点大小
        # Initialize Env
        self.episode_step = 0
        self.goal_pos = np.array([1.5, 1.5])
        self.agent_number = 30
        self.dt = 0.02
        self.second = False
        self.first = True
        # agent 初始位置
        # a = np.array([[3. * np.random.rand() - 1.5, 3. * np.random.rand() - 1.5, 0]]).T
        a = np.array([[6. * np.random.rand() - 3, 6. * np.random.rand() - 3, 0]]).T
        self.initial_conditions = a
        for _ in range(1, self.agent_number):
            self.initial_conditions = np.concatenate((self.initial_conditions, np.array(
                [[2.6 * np.random.rand() - 1.0, 2.6 * np.random.rand() - 1.0, 6.28 * np.random.rand() - 3.14]]).T),
                                                     axis=1)

        self.agents = Robotarium(number_of_robots=self.agent_number, show_figure=False,
                                 initial_conditions=self.initial_conditions, sim_in_real_time=False)
        self.states = self.agents.get_poses().T
        self.agents.step()

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
        u = np.empty([self.agent_number, 2])
        u[0] = action
        for i in range(1, self.agent_number):
            u[i] = np.array([0.15, 0.1])

        self.agents.set_velocities(np.arange(self.agent_number), u.T)
        self.states = self.agents.get_poses().T  # shape[agent_numbe,2]，第一维为RL agent，其他为普通agent
        self.agents.step()
        reward, done, info = self._reward_done()

        self.episode_step += 1

        return self.states, reward, done, info

    def _reward_done(self):
        info = dict()
        reward = 0
        done = False
        dist_goal = self._goal_dist()

        self_state = self.states[0]
        # other_states = self.states[1:]
        other_agent_s = np.delete(self.states[1:], 2, 1)  # 除去欧拉角
        other_s = np.vstack((other_agent_s, self.hazards_locations))
        # # Check boundaries
        # if(self_state[1]>1.9 or self_state[1]<-1.9 or self_state[0]>1.9 or self_state[0]<-1.9):
        #     print('Out of boundaries !!')
        #     reward -= 100
        #     done =True

        for idx in range(np.size(other_s, 0)):
            distSqr = (self_state[0] - other_s[idx][0]) ** 2 + (self_state[1] - other_s[idx][1]) ** 2
            if distSqr < (0.2) ** 2:
                print('Get caught, mission failed !')
                done = True
                reward -= 500

        if self.goal_met():
            print('Reach goal successfully!')
            info['goal_met'] = True
            reward += 500
            done = True

        # # Check if goal is met
        # if self.goal_met() and self.first:
        #     print('Reach goal successfully!')
        #     info['goal_met'] = True
        #     reward += 500
        #     # done = True
        #     self.second = True
        #     self.first = False
        #     self.goal_pos = np.array([-0.5, -0.5])
        # if self.goal_met() and self.second:
        #     print('Reach goal successfully!')
        #     info['goal_met'] = True
        #     reward += 500
        #     # done = True
        #     self.second = False
        #     self.first = False
        # if not self.second and not self.first:
        #     done = True
        else:
            reward -= 0.1 * dist_goal
            # reward += 10 * (self.last_goal_dist - dist_goal)

        self.last_goal_dist = dist_goal

        if self.max_episode_steps <= self.episode_step:
            done = True
        # Include constraint cost in reward
        if np.any(np.sum((self_state[:2] - self.hazards_locations) ** 2, axis=1) < self.hazards_radius ** 2):
            if 'cost' in info:
                info['cost'] += 0.1
            else:
                info['cost'] = 0.1
        # print(reward)
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
        plt.close()

        # a = np.array([[-1.5, -1.5 ,0]]).T
        a = np.array([[6. * np.random.rand() - 3, 6. * np.random.rand() - 3, 0]]).T

        # a = np.array([[3.9 * np.random.rand() - 2.0, 3.9 * np.random.rand() - 2.0, 0]]).T
        self.initial_conditions = a
        for _ in range(1, self.agent_number):
            self.initial_conditions = np.concatenate((self.initial_conditions, np.array(
                [[2.6 * np.random.rand() - 1.0, 2.6 * np.random.rand() - 1.0, 6.28 * np.random.rand() - 3.14]]).T),
                                                     axis=1)
        self.agents = Robotarium(number_of_robots=self.agent_number, show_figure=is_show_figure,
                                 initial_conditions=self.initial_conditions, sim_in_real_time=False)
        if (is_show_figure):
            self.agents.axes.add_patch(
                patches.Circle(self.hazards_locations[0], self.hazards_radius - 0.05, fill=True))  # Obstacle 1
            self.agents.axes.add_patch(
                patches.Circle(self.hazards_locations[1], self.hazards_radius - 0.05, fill=True))  # Obstacle 2
            self.agents.axes.add_patch(
                patches.Circle(self.hazards_locations[2], self.hazards_radius - 0.05, fill=True))  # Obstacle
            self.agents.axes.add_patch(patches.Circle(self.goal_pos, self.goal_size, fill=False, zorder=10))  # target

        self.states = self.agents.get_poses().T
        self.agents.step()

        self.episode_step = 0
        other_agent_s = np.delete(self.states[1:], 2, 1)  # 除去欧拉角
        other_s = np.vstack((other_agent_s, self.hazards_locations))

        # Re-initialize last goal dist
        self.last_goal_dist = self._goal_dist()

        return self.get_obs(), other_s

    def render(self):
        """Render the environment to the screen

         Parameters
         ----------
         mode : str
         close : bool

         Returns
         -------

         """
        a = np.array([[2. * np.random.rand() - 0.5, 2. * np.random.rand() - 0.5, 0]]).T
        self.initial_conditions = a
        for _ in range(1, self.agent_number):
            self.initial_conditions = np.concatenate((self.initial_conditions, np.array(
                [[2.6 * np.random.rand() - 1.0, 2.6 * np.random.rand() - 1.0, 6.28 * np.random.rand() - 3.14]]).T),
                                                     axis=1)

        self.agents = Robotarium(number_of_robots=self.agent_number, show_figure=True,
                                 initial_conditions=self.initial_conditions, sim_in_real_time=False)
        # Draw barrier
        self.agents.axes.add_patch(
            patches.Circle(self.hazards_locations[0], self.hazards_radius, fill=True))  # Obstacle 1
        self.agents.axes.add_patch(
            patches.Circle(self.hazards_locations[1], self.hazards_radius, fill=True))  # Obstacle 2
        self.agents.axes.add_patch(
            patches.Circle(self.hazards_locations[2], self.hazards_radius, fill=True))  # Obstacle
        self.agents.axes.add_patch(patches.Circle(self.goal_pos, self.goal_size, fill=False, zorder=10))  # target

        # self.states = self.agents.get_poses().T
        # self.agents.step()
        # self.agents.get_poses()

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


def get_random_hazard_locations(n_hazards, hazard_radius, bds=None):
    """

    Parameters
    ----------
    n_hazards : int
        Number of hazards to create
    hazard_radius : float
        Radius of hazards
    bds : list, optional
        List of the form [[x_lb, x_ub], [y_lb, y_ub] denoting the bounds of the 2D arena

    Returns
    -------
    hazards_locs : ndarray
        Numpy array of shape (n_hazards, 2) containing xy locations of hazards.
    """

    if bds is None:
        bds = np.array([[-3., -3.], [3., 3.]])

    # Create buffer with boundaries
    buffered_bds = bds
    buffered_bds[0] += hazard_radius
    buffered_bds[1] -= hazard_radius

    hazards_locs = np.zeros((n_hazards, 2))

    for i in range(n_hazards):
        successfully_placed = False
        iter = 0
        while not successfully_placed and iter < 500:
            hazards_locs[i] = (bds[1] - bds[0]) * np.random.random(2) + bds[0]
            successfully_placed = np.all(np.linalg.norm(hazards_locs[:i] - hazards_locs[i], axis=1) > 3 * hazard_radius)
            iter += 1

        if iter >= 500:
            raise Exception('Could not place hazards in arena.')

    return hazards_locs
