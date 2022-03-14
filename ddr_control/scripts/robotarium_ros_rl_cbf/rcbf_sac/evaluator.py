import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from rcbf_sac.utils import *


class Evaluator(object):

    def __init__(self, num_episodes, interval, save_path=''):
        self.num_episodes = num_episodes
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes, 0)

    def __call__(self, env, policy, cbf_wrapper=None, dynamics_model=None, debug=False, visualize=False, save=True):

        self.is_training = False
        result = []
        self.rewards = []
        all_states = []
        all_o_states = []
        all_actions = []
        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation, other_s = env.reset(True)

            # Make sure to start from a safe state
            if cbf_wrapper and dynamics_model:
                out = None
                while out is None or cbf_wrapper.get_min_h_val(out) < 1e-4:
                    observation, other_s = env.reset(True)
                    state = dynamics_model.get_state(observation)
                    out = dynamics_model.get_output(state)

            episode_steps = 0
            episode_reward = 0.
            episode_state = []
            episode_o_state = []
            episode_action = []
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation, other_s)

                observation, other_s, reward, done, info = env.step(action)
                state = dynamics_model.get_state(observation)

                # if visualize:
                #     env.render()

                # update
                episode_reward += reward
                episode_steps += 1
                episode_state.append(state)
                episode_o_state.append(other_s)
                episode_action.append(action)

            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode, episode_reward))
            result.append(episode_reward)
            all_actions.append(episode_action)
            all_states.append(episode_state)
            all_o_states.append(episode_o_state)
        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result])
        self.rewards.append(episode_reward)
        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
            savemat('{}/validate_'.format(self.save_path) + 'data.mat', {'actions': all_actions,
                                                                         'states': all_states,
                                                                         'o_states': all_o_states})
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error = np.std(self.results, axis=0)

        x = range(0, self.results.shape[1] * self.interval, self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn + '.png')
        savemat(fn + '.mat', {'reward': self.results})
        savemat(fn + '_01.mat', {'reward': self.rewards})
        plt.close()
