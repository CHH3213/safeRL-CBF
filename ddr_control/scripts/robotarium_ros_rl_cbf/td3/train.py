#!/usr/bin/python3
# --*--coding:utf-8--*--
"""
==========IEEE_TAI_DDR-v2===============
@File: train.py
@Time: 2022/3/16 下午4:53
@Author: chh3213
@Description: 换乘episodes的形式
========Above the sun, full of fire!=============
"""
from comet_ml import Experiment
from td3 import utils
from td3.replay_memory import ReplayMemory
import numpy as np
from tqdm import tqdm
import scipy.io as sio
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../new_envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../envs'))
sys.path.append('..')

from controllers.simple_pv import *


def train(env, args, policy, file_name):
    if args.load_model != "" and args.restore:
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./td3/models/{policy_file}")

    # Create an experiment with your api key
    # project_name = 'TAI-td3-ros-Robotarium-environment'

    if args.use_cbf:
        project_name = 'cbf-layer-environment-td3'
    else:
        project_name = 'no_cbf-layer-environment-td3'
    if args.log_comet:
        experiment = Experiment(
            api_key=args.comet_key,
            project_name=project_name,
            workspace=args.comet_workspace,
        )
        # Log args on comet.ml
        experiment.log_parameters(vars(args))
        experiment_tags = [str(args.batch_size) + '_batch']
        experiment.add_tags(experiment_tags)
        experiment.log_parameters({'obs_dim':env.observation_space.shape[0]})
    else:
        experiment=None
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    replay_buffer = ReplayMemory(args.replay_size, args.seed)

    N = env.agent_number
    index = 0  # agent 索引
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    all_rewards = []
    all_actions = []
    all_states_save = []
    reward_st = np.array([0])
    t = 0
    for i_episode in tqdm(range(args.max_episodes)):
        steps_reward = []
        steps_action = []
        steps_state = []

        states = env.reset(False)
        obs,other_s = env.get_obs(index)
        actions = np.zeros((N, 2))
        actions[:, 0] = 0.15  # 线速度恒定
        done = False
        while not done:
            t += 1
            episode_timesteps += 1

            action = (policy.select_action(obs,other_s)+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)
            # print(action)
            for idx in range(0, N):
                if idx == index:
                    continue

                actions[idx, 0] = 0.15 * 0.85 # 线速度恒定
                actions[idx, 1] = np.random.uniform(-1,1)
                # actions[idx, 0] = 0.15


            safe_a = action            
            actions[index] = safe_a
            # Perform action
            next_states = env.step(actions)
            reward, done, info = env._reward_done(index)
            next_obs,other_s = env.get_obs(index)
            done_bool = float(done)

            # Store data in replay buffer
            # print(np.shape(obs),np.shape(other_s),np.shape(action),np.shape(reward),np.shape(next_obs),np.shape(done_bool))
            replay_buffer.push(obs,other_s, action, reward, next_obs, done_bool)

            states = next_states
            obs = next_obs
            episode_reward += reward
            steps_reward.append(reward)
            steps_action.append(action)
            steps_state.append(states)

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)
            else:
                critic_loss, actor_loss = 0, 0
            if experiment:
                # Comet.ml logging
                experiment.log_metric('loss/critic', critic_loss, step=t)
                experiment.log_metric('loss/actor', actor_loss, step=t)
        if experiment:
            experiment.log_metric('reward/train', episode_reward, step=i_episode)

        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(
            f"Total Episode: {i_episode}  Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

        reward_st = np.append(reward_st, episode_reward)

        all_rewards.append(steps_reward)
        all_actions.append(steps_action)
        all_states_save.append(steps_state)

        np.save(f"./td3/results/{file_name}/train_episode_reward", reward_st)

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        if i_episode % args.save_freq == 0:
            policy.save(f"./td3/models/{file_name}")
            sio.savemat(f"./td3/results/{file_name}" + '/train_data.mat', 
            {
                'rewards': all_rewards,
                'actions': all_actions,
                'states': all_states_save,
                'collison_obstacle': env.collison_obstacle,
                'get_caught': env.get_caught
            })
