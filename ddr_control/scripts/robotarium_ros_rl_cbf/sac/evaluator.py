import copy
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from sac.utils import *
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../new_envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
sys.path.append('..')
from controllers.simple_pv import *



class Evaluator(object):

    def __init__(self, args,num_episodes, interval, seed, use_cbf=True, save_path=''):
        self.args = args
        self.num_episodes = num_episodes
        self.interval = interval
        self.save_path = save_path
        self.seed = seed
        self.use_cbf = use_cbf
        self.results = np.array([]).reshape(num_episodes, 0)

    def __call__(self, env, policy, index_list=[0], debug=False, visualize=False, save=True):

        self.is_training = False
        result = []
        self.rewards = []
        all_states = []
        all_o_states = []
        all_actions = []
        all_origin_actions = []
        all_dones = []
        all_success = []
        even_success = []
        odd_success = []
        judge_success = []
        judge_success_2 = []
        all_steps = []
        N = env.agent_number
        for episode in range(self.num_episodes):

            # reset at the start of episode
            episode_steps = 0
            episode_state = []
            steps_origin_action = []
            episode_o_state = []
            episode_action = []
            states = env.reset(True)
            observations = []
            others_s=[]
            for index in index_list:
                observation,other_s = env.get_obs(index)
                observations.append(observation)
                others_s.append(other_s)
            episode_reward = 0
            # start episode
            dones = [False]
            is_all_success = [False]
            is_even_success = [False]
            is_odd_success = [False]
            is_judge_success = [False]
            is_judge_success = [False]
            is_judge_success_2 = [False]
            is_all_success_2 = [False]
            actions = np.zeros((N, 2))
            actions[:, 0] = 0.15  # 线速度恒定
            while not any(dones) and (not all(is_judge_success_2)) and (not all(is_all_success)):
            # while (not any(dones)) and (not all(is_judge_success)):
                start = time.time()
                # if env.episode_step%5==0:
                #     last_states = copy.deepcopy(states)
                # print('last_states',last_states)
                # print('states',states)
                # basic operation, action ,reward
                action_list = []
                for idx, observation,other_s in zip(index_list, observations,others_s):
                    action = policy(observation,other_s)
                    action_list.append(action)

                idx_all = [i for i in range(0, N)]
                idx_left = np.delete(idx_all, index_list, 0)
                for idx in idx_left:
                    pursuit_target = self.compute_nearest(states[idx], states[index_list])
                    actions[idx, 1] = simple_pursuit(states[idx], pursuit_target)
                    # actions[idx, :] = Simple_Catcher(pursuit_target, states[idx])
                    # actions[idx, 1] = 0.1
                    actions[idx, 0] = 0.15 * 1.0
                    # actions[idx, 1] = np.random.uniform(-1, 1)


                for i, index in enumerate(index_list):
                    safe_a = action_list[i]
                    add_rand = np.random.normal(0, 1 ** 0.5, safe_a.shape).clip(-0.5,0.5)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
                    add_rand = 0
                    safe_a = safe_a + add_rand
                    # 干扰
                    if self.args.disturbance:
                        noise = np.random.normal(0, 1 ** 0.5, safe_a.shape).clip(-1,1)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
                        # safe_a = (safe_a+noise).clip(-2.5,2.5)
                        safe_a = safe_a+noise

                    actions[index,1] = safe_a
                temp = time.time()
                next_states = env.step(actions)
                # print('step_time', time.time() - temp)
                dones = []
                is_all_success = []
                is_even_success = []
                is_odd_success = []
                is_all_success = []
                is_all_success_2 = []
                for index in index_list:
                    # print(index)
                    reward, done, info = env._reward_done(index=index)
                    is_success, is_success_2 = env._is_success()
                    is_all_success.append(is_success)

                    is_all_success_2.append(is_success_2)

                    dones.append(done)
                    # if index%2==0:
                    #     is_even_success.append(is_success)
                    # elif index%2 ==1:
                    #     is_odd_success.append(is_success)
            if env.agent_number==10:
                is_judge_success = [any(is_all_success[0:2]),any(is_all_success[2:4]),any(is_all_success[4:6]),any(is_all_success[6:8])]
                is_judge_success_2 = [any(is_all_success_2[0:2]), any(is_all_success_2[2:4]), any(is_all_success_2[4:6]), any(is_all_success_2[6:8])]

            elif env.agent_number==3:
                is_judge_success = [any(is_all_success[0:2])]
                is_judge_success_2 = [any(is_all_success_2[0:2])]

                # print(dones)
                # print('is_odd_success: ',is_odd_success)
                # print('is_even_success: ',is_even_success)
                # print(is_all_success)
                # print(is_judge_success)
                # print(states)
                observations = []
                others_s = []
                for index in index_list:
                    observation,other_s = env.get_obs(index)
                    observations.append(observation)
                    others_s.append(other_s)
                states = next_states
                episode_reward += reward
                episode_steps += 1
                episode_state.append(copy.deepcopy(states))
                # print(episode_state)
                episode_action.append(copy.deepcopy(actions[:,1]))
                steps_origin_action.append(copy.deepcopy(action_list))
                if (time.time() - start) + 1e-5 < env.dt:
                    time.sleep(np.abs(env.dt - (time.time() - start)))
                # print("step time: ", time.time() - start)

            print('episode',episode, "episode steps:", env.episode_step)
            if debug: prYellow('[Evaluate] #Episode{}: episode_reward:{}'.format(episode, episode_reward))
            # result.append(episode_reward)
            all_actions.append(episode_action)
            all_origin_actions.append(steps_origin_action)
            all_states.append(episode_state)
            all_dones.append(dones)
            all_success.append(is_all_success)
            judge_success.append(is_judge_success)
            judge_success_2.append(is_judge_success_2)
            even_success.append(is_even_success)
            judge_success.append(is_judge_success)
            odd_success.append(is_odd_success)
            all_steps.append(env.episode_step)
            # result = np.array(result).reshape(-1, 1)
            # self.results = np.hstack([self.results, result])
            # self.rewards.append(episode_reward)
        if save:
            # self.save_results('{}/validate_reward'.format(self.save_path))
            str_index = str(index_list)[1:-1]
            str_index = str_index.replace(",", "")
            str_index = str_index.replace(" ", "")
            save_test = self.save_path+'/test_episodes_'+str(self.num_episodes)
            if not os.path.exists(save_test):
                os.makedirs(save_test)
            savemat('{}/agent_{}_cbf_{}_disturb_{}_delay_{}'.format(save_test,env.agent_number,self.use_cbf,self.args.disturbance,self.args.time_delay) + '_seed_{}_good_{}_{}.mat'.format(self.seed, str_index,time.strftime( "%Y-%m-%d-%H-%M-%S",
                                                                                           time.localtime())),
                    { 'origin_actions':all_origin_actions,
                     'actions': all_actions,
                     'states': all_states,
                     'collision_or_max_steps':all_dones,
                     'is_all_success':all_success,
                     'judge_success':judge_success,
                      'judge_success_2': judge_success_2,
                     'episode_steps':all_steps,
                     'max_episode_steps':env.max_episode_steps,
                    'collison_obstacle': env.collison_obstacle,
                    'get_caught': env.get_caught
                     })
        return np.mean(result)

    def compute_nearest(self, self_state, other_states):
        """
        compute nearest angent is which"""
        dist_list = []
        for other_s in other_states:
            distSqr = (self_state[0] - other_s[0]) ** 2 + (self_state[1] - other_s[1]) ** 2
            dist_list.append(distSqr)
        avoid_index = np.argsort(dist_list)[0]
        return other_states[avoid_index]

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



def test(agent, env, index, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)

    def policy(observation, other_s):
        action = agent.select_action(observation,other_s, evaluate=True)
        return action

    evaluate(env, policy, index, debug=debug, visualize=visualize, save=True)