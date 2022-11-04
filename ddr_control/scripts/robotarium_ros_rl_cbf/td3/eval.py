#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========IEEE_TAI_DDR-v2===============
@File: eval.py
@Time: 2022/3/16 下午4:53
@Author: chh3213
@Description: 这个备份文件使用的CBF是将所有智能体和障碍物都考虑进去，并不是通过最近的障碍物和最近的智能体创建cbf
========Above the sun, full of fire!=============
"""
# 多进程
from concurrent.futures import thread
from threading import Thread, Lock

from comet_ml import Experiment
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import os
import sys
import copy
import torch
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '../new_envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
sys.path.append('..')
from controllers.simple_pv import *


global threads

def eval(env, args, policy, file_name, index_list=[0]):
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./td3/models/{policy_file}")
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    N = env.agent_number
    index = 0  # agent 索引

    all_origin_actions = []
    all_actions = []
    all_states_save = []
    all_success = []
    judge_success = []
    judge_success_2 = []
    all_steps = []
    all_dones = []
    # 定义路径
    str_index = str(index_list)[1:-1]
    str_index = str_index.replace(",", "")
    str_index = str_index.replace(" ", "")
    if not os.path.exists(f"./td3/results/{file_name}/episodes_{args.eval_episodes}"):
        os.makedirs(f"./td3/results/{file_name}/episodes_{args.eval_episodes}")

    for t in tqdm(range(args.eval_episodes)):
        steps_action = []
        steps_state = []
        steps_origin_action = []
        episode_reward = 0
        episode_timesteps = 0

        states, dones = env.reset(args.render), [False]
        observations = []
        others_s=[]
        for index in index_list:
            obs,other_s = env.get_obs(index)
            observations.append(obs)
            others_s.append(other_s)

        actions = np.zeros((N, 2))
        actions[:, 0] = 0.15  # 线速度恒定      
        is_judge_success=[False]
        is_judge_success_2=[False]
        is_all_success = [False]
        is_all_success_2 = [False]
        # while not any(dones) and (not all(is_judge_success)):
        while not any(dones) and (not all(is_judge_success_2)) and (not all(is_all_success)):
            start = time.time()
            episode_timesteps += 1
            if env.episode_step%5==0:  # 3个agents
            # if env.episode_step%2==0:  # 10个agents
            # if env.episode_step%1==0:
                last_states = copy.deepcopy(states)
            # last_states = copy.deepcopy(states)
            action_list = []
            for idx, observation,other_s in zip(index_list, observations,others_s):
                action = policy.select_action(obs,other_s)
                action_list.append(action)

            # print(action)
            ### 所有障碍物都输进去
            if args.time_delay:
                all_states_ = np.vstack((last_states, env.hazards_locations))

                # all_states_ = np.vstack((last_states, env.nearest_loca))
            else:
                all_states_ = np.vstack((states, env.hazards_locations))
                # all_states_ = np.vstack((states, env.nearest_loca))
            idx_all = [i for i in range(0, N)]
            idx_left = np.delete(idx_all, index_list, 0)
            for idx in idx_left:
                pursuit_target = compute_nearest(states[idx], states[index_list])
                actions[idx, 1] = simple_pursuit(states[idx], pursuit_target)
                
                # actions[idx, 1] = CBF([actions[idx, 1]], all_states, idx)
                actions[idx, 0] = 0.15
                # actions[idx, 1] = np.random.uniform(-1, 1)

                

            for i, index in enumerate(index_list):
                # 只考虑最近的障碍物和最近的智能体
                # if args.time_delay:
                #     # all_states_ = np.vstack((last_states, env.nearest_obstacle(index)))
                #     all_states_ = np.vstack((last_states, env.nearest_obstacle(index)))

                # else:
                #     all_states_ = np.vstack((states, env.nearest_obstacle(index)))
                
                safe_a = action_list[i]
                safe_a = safe_a   
                # if safe_a != action_list[i]:
                #     print(all_states_) 
                #     #  [-0.70368036 -0.16339836  1.07789404]
                #     # [-0.22515132  0.5010884  -2.53232707] 距离大概为0.8
                #     break               # 干扰
                if args.disturbance:
                        # noise = np.random.rand()-0.5
                        noise = np.random.normal(0, 1 ** 0.5, safe_a.shape).clip(-1,1)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
                        # noise = np.random.normal(0, 1 ** 0.5, safe_a.shape)#创建一个均值为mean，方差为var呈高斯分布的图像矩阵
                        # print(noise)
                        # print(safe_a)
                        # safe_a = (safe_a+noise).clip(-2.5,2.5)
                        safe_a = safe_a+noise
                        # print(safe_a)
                        
                actions[index,1] = safe_a
                    

            
            # Perform action
            # s = time.time()

            next_states = env.step(actions)
            dones = []
            is_all_success = []
            is_all_success_2 = []

            for index in index_list:
                reward, done, info = env._reward_done(index=index)
                is_success, is_success_2 = env._is_success()
                # print(is_success_2)
                is_all_success.append(is_success)

                is_all_success_2.append(is_success_2)
                dones.append(done)
            # print(dones)
            if env.agent_number==10:
                is_judge_success = [any(is_all_success[0:2]),any(is_all_success[2:4]),any(is_all_success[4:6]),any(is_all_success[6:8])]
                is_judge_success_2 = [any(is_all_success_2[0:2]), any(is_all_success_2[2:4]), any(is_all_success_2[4:6]), any(is_all_success_2[6:8])]

            elif env.agent_number==3:
                is_judge_success = [any(is_all_success[0:2])]
                is_judge_success_2 = [any(is_all_success_2[0:2])]


            observations = []
            others_s = []
            for index in index_list:
                observation,other_s = env.get_obs(index)
                observations.append(observation)
                others_s.append(other_s)
            states = next_states
            episode_reward += reward
            steps_action.append(copy.deepcopy(actions[:, 1]))
            steps_state.append(copy.deepcopy(states))
            steps_origin_action.append(copy.deepcopy(action_list))

            # print(is_judge_success)
            # print(is_judge_success_2)
            if (time.time() - start) + 1e-5 < env.dt:
                time.sleep(np.abs(env.dt - (time.time() - start)))
            # print(time.time()-start)

        all_actions.append(steps_action)
        all_origin_actions.append(steps_origin_action)
        all_states_save.append(steps_state)
        all_success.append(is_all_success)
        judge_success.append(is_judge_success)
        judge_success_2.append(is_judge_success_2)
        all_steps.append(env.episode_step)
        all_dones.append(dones)

        # if (t+1)%10==0:

        #     sio.savemat(f"./results/{file_name}/episodes_{args.eval_episodes}" + '/gazebo_{}_agent_{}_cbf_{}_disturb_{}_delay_{}_seed_{}_good_{}_{}.mat'.format(
        #         args.ros_env, env.agent_number,args.use_cbf,args.disturbance,args.time_delay,args.seed,str_index,time.strftime( "%Y-%m-%d-%H-%M",time.localtime())), 
        #             {
        #                 'origin_actions': all_origin_actions,
        #                 'actions': all_actions,
        #                 'states': all_states_save, 
        #                 'judge_success':judge_success, 
        #                 'judge_success_2':judge_success_2, 
        #                 'is_all_success':all_success,
        #                 'collision_or_max_steps':all_dones, 
        #                 'max_episode_steps':env.max_episode_steps,
        #                 'episode_steps':all_steps,
        #                 'collison_obstacle': env.collison_obstacle,
        #                 'get_caught': env.get_caught
        #             })
    sio.savemat(f"./td3/results/{file_name}/episodes_{args.eval_episodes}" + '/gazebo_{}_agent_{}_cbf_{}_disturb_{}_delay_{}_seed_{}_good_{}_{}.mat'.format(
                    args.ros_env, env.agent_number,args.use_cbf,args.disturbance,args.time_delay,args.seed,str_index,time.strftime( "%Y-%m-%d-%H-%M",time.localtime())), 
                        {
                            'origin_actions': all_origin_actions,
                            'actions': all_actions,
                            'states': all_states_save, 
                            'judge_success':judge_success, 
                            'judge_success_2':judge_success_2, 
                            'is_all_success':all_success,
                            'collision_or_max_steps':all_dones, 
                            'max_episode_steps':env.max_episode_steps,
                            'episode_steps':all_steps,
                            'collison_obstacle': env.collison_obstacle,
                            'get_caught': env.get_caught
                        })


def compute_nearest(self_state, other_states):
    dist_list = []
    for other_s in other_states:
        distSqr = (self_state[0] - other_s[0]) ** 2 + (self_state[1] - other_s[1]) ** 2
        dist_list.append(distSqr)
    avoid_index = np.argsort(dist_list)[0]
    return other_states[avoid_index]


