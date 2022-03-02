# -*- coding: utf-8 -*-
import rospy
import gym
from gym.spaces import Box
import numpy as np
from CBF_TD3_Agent01 import TD3 as TD3_Agent01
from CBF_TD3_Agent01 import ReplayBuffer as ReplayBuffer_Agent01
from CBF_TD3_Agent02 import TD3 as TD3_Agent02
from CBF_TD3_Agent02 import ReplayBuffer as ReplayBuffer_Agent02
from CBF_TD3_Agent03 import TD3 as TD3_Agent03
from CBF_TD3_Agent03 import ReplayBuffer as ReplayBuffer_Agent03
import argparse
import os
import tensorflow as tf
import time
import scipy.io as sio
import sys
import random

np.set_printoptions(suppress=True)  # 取消科学计数法输出
sys.path.append('..')

from controllers.simple_pv import simple_pursuit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--episodes', default=3500, type=int)
    parser.add_argument("--max_steps", type=int, default=500, help="maximum max_steps length")  # 每个episode的步数为400步
    parser.add_argument('--saveData_dir', default="./save/td3_add_obs_{}/data".format(
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())),
                        help="directory to store all experiment data")
    parser.add_argument('--testData_dir', default="./save/td3_add_obs_2021-11-30-18:26:30/data",
                        help="directory to store all experiment data")
    parser.add_argument('--saveModel_dir', default='./save/td3_add_obs_{}/models'.format(
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())),
                        help="where to store/load network weights")
    parser.add_argument('--load_dir', default='./save/td3_add_obs_2022-01-10-19:43:09/models',
                        help="where to load network weights")

    parser.add_argument('--checkpoint_frequency', default=10, type=int,
                        help="how often to checkpoint")
    parser.add_argument('--testing', default=False, action="store_true",
                        help="reduces exploration substantially")
    parser.add_argument("--restore", default=True, action="store_true")
    parser.add_argument('--memory_size', default=100000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0:4], device_type='GPU')

    frame_idx = 0
    ######################Hyperameters#########################
    EXPLORE_STEPS = 100
    EXPLORE_NOISE_SCALE = 1.0  # range of action noise for exploration
    UPDATE_ITR = 2  # repeated updates for single step
    EVAL_NOISE_SCALE = 0.5  # range of action noise for evaluation of action value
    REWARD_SCALE = 1.  # value range of reward
    #################################################################
    SOFT_Q_LR = 3e-4  # q_net learning rate
    POLICY_LR = 3e-4  # policy_net learning rate
    ALPHA_LR = 3e-4  # alpha learning rate
    AUTO_ENTROPY = True  # automatically updating variable alpha for entropy
    ##############################################################3
    if not args.testing:
        from envs.train_safe_env import TAI_Env
        env = TAI_Env()
    else:
        from envs.test_safe_env import TAI_Env
        env = TAI_Env()

    # env = gym.make('Pendulum-v0')
    max_steps = args.max_steps  # max_steps per episode
    assert isinstance(env.observation_space[0], Box), "observation space must be continuous"
    assert isinstance(env.action_space[0], Box), "action space must be continuous"

    # Randomly initialize critic,actor,target critic, target actor network  and replay buffer

    HIDDEN_DIM = 128
    POLICY_TARGET_UPDATE_INTERVAL = 2  # delayed steps for updating the policy network and target networks
    replay_Buffer = [ReplayBuffer_Agent01(args.memory_size), ReplayBuffer_Agent02(args.memory_size),
                     ReplayBuffer_Agent03(args.memory_size)]

    s_dim = env.observation_space[0].shape[0]
    a_dim = env.action_space[0].shape[0]
    a_bound = env.action_space[0].high
    agent_1 = TD3_Agent01(s_dim, a_dim, a_bound, HIDDEN_DIM, replay_Buffer[0], POLICY_TARGET_UPDATE_INTERVAL,
                          args.learning_rate, args.learning_rate)
    agent_2 = TD3_Agent02(s_dim, a_dim, a_bound, HIDDEN_DIM, replay_Buffer[0], POLICY_TARGET_UPDATE_INTERVAL,
                          args.learning_rate, args.learning_rate)
    agent_3 = TD3_Agent03(s_dim, a_dim, a_bound, HIDDEN_DIM, replay_Buffer[0], POLICY_TARGET_UPDATE_INTERVAL,
                          args.learning_rate, args.learning_rate)
    agents = [agent_1, agent_2, agent_3]
    reward_per_episode = 0
    safe_per_episode = 0
    total_reward = 0
    total_safe = 0
    print("Number of Steps per episode:", max_steps)
    # saving reward:
    reward_st = np.array([0])
    reward_st_ddr0 = np.array([0])
    reward_st_ddr1 = np.array([0])
    safe_st = np.array([0])
    count_st = np.array([0])
    start_time = time.time()
    # 存储数据列表
    reward_each_episode = []
    action_each_episode = []
    state_each_episode = []
    safe_each_episode = []
    count_each_episode = []
    count_wall = 0
    count_agent = 0
    VAR = 1
    # 目标值
    state = env.reset()
    state = np.array(state)
    state = state.astype(np.float32)
    for i, agent in enumerate(agents):
        agent.policy_net([state[i]])
        agent.target_policy_net([state[i]])
        if args.restore:
            agent.load(args.load_dir, env.ddr_name[i])
    t_state = time.time()
    save_txt_dir = './save_txt/txt_{}'.format(
        time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    if not os.path.exists(save_txt_dir):
        os.makedirs(save_txt_dir)
    reward_txt_name = save_txt_dir + '/td3_qp_reward_st_{}.txt'.format(
        time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
    reward_txt_name_ddr0 = save_txt_dir + '/td3_qp_reward_st_ddr0_{}.txt'.format(
        time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
    reward_txt_name_ddr1 = save_txt_dir + '/td3_qp_reward_st_ddr1_{}.txt'.format(
        time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
    count_txt_name = save_txt_dir + '/td3_qp_count_st_{}.txt'.format(
        time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
    if not args.testing:
        print('train!!!!!!!!!!!!!!!!')
        for episode in range(args.episodes):
            print("==== Starting episode no:", episode, "====", "\n")
            r = rospy.Rate(5)
            observation = env.reset()
            observation = np.array(observation)
            observation = observation.astype(np.float32)
            # observation = s_norm(env,observation)
            # observation = np.array(observation)
            reward_per_episode = 0  # 用来存储一个episode的总和
            reward_per_episode_ddr0 = 0  # 用来存储一个episode的总和
            reward_per_episode_ddr1 = 0  # 用来存储一个episode的总和
            safe_per_episode = 0  # 用来存储一个episode的总和
            count_per_episode = 0
            safe_per_step = 0
            # 存储每个episode的数据
            reward_steps = []
            action_steps = []
            obs_steps = []
            safe_steps = []
            count_steps = []
            steps = 0
            start = time.time()
            # reward_st = np.array([0])
            actions = [[0], [0], [0]]
            for t in range(max_steps):
                # rendering environmet (optional)
                state = observation
                # print(state)
                for i, agent in enumerate(agents):
                    action = agent.policy_net.get_action(actions, state[i], EXPLORE_NOISE_SCALE)
                    # if frame_idx > EXPLORE_STEPS:
                    #     action = agent.policy_net.get_action(state[i], EXPLORE_NOISE_SCALE)
                    #     # print('train!!')
                    # else:
                    #     action = agent.policy_net.sample_action()
                    # action = 0.2*i
                    # action = action * 2
                    actions[i] = action

                """随机一个智能体变为追击离它最近的agent"""
                "==================================="
                bad_index = 2
                dist = [np.linalg.norm(state[i][0:2] - state[bad_index][0:2]) for i in range(env.agent_number)]
                avoid_index = np.argsort(dist)[1]  # 返回第二小的索引
                # print('bad_index', bad_index)
                a_bad = simple_pursuit(state[bad_index], state[avoid_index])
                a_bad = np.array([a_bad])
                a_bad = a_bad.astype(np.float32)
                actions[bad_index] = a_bad
                "========================================="
                print("Action at step", t, " :", actions, "\n")

                observation, rewards, done_n, safe, count, info = env.step(actions)
                # print(safe)
                # print('###################################')
                # print(np.array(safe[0:2]).flatten())
                # # print(safe)
                # print('######################################')
                observation = np.array(observation)
                observation = observation.astype(np.float32)  # 转换数组的类型
                # print('observation', observation)
                # 每个episode、每一步的数据
                target = np.array([0, 0])
                # rewards = np.array(rewards)
                # rewards = rewards.astype(np.float32)
                # for i in enumerate(agents):
                rewards = rewards + env.rewardshaping(state, observation, target, df=0.8)
                count_steps.append(count)
                reward_steps.append(rewards)
                action_steps.append(actions)
                obs_steps.append(observation)
                safe_steps.append(safe)
                # print(reward)
                # add s_t,s_t+1,action,reward to experience memory
                for i, agent in enumerate(agents):
                    # print(i)
                    replay_Buffer[i].push(state[i], actions[i], rewards[i], observation[i], done_n[i], safe[i])
                    # print(state[i])
                    # print('#######################0')
                    # print(actions[i])
                    # print('#############################1')
                    # print(rewards[i])
                    # print('#############################2')
                    # print(observation[i])
                    # print('#############################3')
                    # print(done_n[i])
                    # print('#############################4')
                    # print(safe[i])
                    # print('#############################5')
                    if len(replay_Buffer[i]) > args.batch_size:
                        # s = time.time()
                        for j in range(UPDATE_ITR):
                            # train critic and actor network
                            if i != 2:
                                agent.update(actions, args.batch_size, EVAL_NOISE_SCALE, REWARD_SCALE)
                        # print(time.time()-s)

                reward_per_episode += np.sum(rewards)
                reward_per_episode_ddr0 += np.sum(rewards[0])
                reward_per_episode_ddr1 += np.sum(rewards[1])
                count_per_episode += count
                safe_per_episode += np.sum(safe)
                frame_idx += 1
                # check if episode ends:
                # print(t)
                print(done_n)
                if env.wall_collide:
                    count_wall += 1
                if env.agent_colliside:
                    count_agent += 1
                # print(any(done_n))
                if any(done_n) or (t == max_steps - 1):
                    # print('EPISODE:  ', episode, ' Steps: ', t, ' ddr0 Reward: ', reward_per_episode_ddr0,
                    #       ' ddr1 Reward: ', reward_per_episode_ddr1,' Total Reward: ', reward_per_episode)
                    # print("Printing reward to file")
                    # exploration_noise.reset() #reinitializing random noise for action exploration
                    reward_st = np.append(reward_st, reward_per_episode)
                    reward_st_ddr0 = np.append(reward_st_ddr0, reward_per_episode_ddr0)
                    reward_st_ddr1 = np.append(reward_st_ddr1, reward_per_episode_ddr1)
                    safe_st = np.append(safe_st, safe_per_episode)
                    count_st = np.append(count_st, count_per_episode)

                    reward_each_episode.append(reward_steps)
                    action_each_episode.append(action_steps)
                    state_each_episode.append(obs_steps)
                    safe_each_episode.append(safe_steps)
                    count_each_episode.append(count_steps)
                    np.savetxt(reward_txt_name_ddr0, reward_st_ddr0, newline="\n")
                    np.savetxt(reward_txt_name_ddr1, reward_st_ddr1, newline="\n")
                    np.savetxt(reward_txt_name, reward_st, newline="\n")
                    np.savetxt(count_txt_name, count_st, newline="\n")
                    # np.savetxt(safe_txt_name, safe_st, newline="\n")
                    print('count_wall', count_wall)
                    print('count_agent', count_agent)
                    print('\n\n')
                    steps = t
                    break
                r.sleep()
                # print('a step time__', time.time() - start)

            if (episode + 1) % args.checkpoint_frequency == 0:
                if not os.path.exists(args.saveModel_dir):
                    os.makedirs(args.saveModel_dir)

                for i, agent in enumerate(agents):
                    agent.save(args.saveModel_dir, env.ddr_name[i])

            # if episode % 200 == 0:
            #     model_dir = './save/td3_safe_temp_{}/models'.format(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))
            #     if not os.path.exists(model_dir):
            #         os.makedirs(model_dir)
            #     for i, agent in enumerate(agents):
            #         agent.save(model_dir, env.ddr_name[i])

            # print("saving model to {}".format(args.saveModel_dir))
            print("steps: {}, episodes: {}, episode reward: {}, ddr0 Reward:{}, ddr1 Reward:{}, time: {} \n".format(
                steps, episode, reward_per_episode, reward_per_episode_ddr0, reward_per_episode_ddr1,
                round(time.time() - start_time, 3)))
            if not os.path.exists(args.saveData_dir):
                os.makedirs(args.saveData_dir)
            sio.savemat(args.saveData_dir + '/data.mat', {'episode_reward': reward_each_episode,
                                                          'episode_action': action_each_episode,
                                                          'episode_state': state_each_episode,
                                                          'episode_safe': safe_each_episode,
                                                          'episode_count': count_each_episode,
                                                          'count_wall': count_wall,
                                                          'count_agent': count_agent})
            print('count_wall', count_wall)
            print('count_agent', count_agent)
        # 保存数据
        total_reward += reward_per_episode
        total_safe += safe_per_episode
        print("Average reward per episode {}".format(total_reward / args.episodes))
        print("Average safe per episode {}".format(total_safe / args.episodes))
        print("Finished {} episodes in {} seconds".format(args.episodes,
                                                          time.time() - start_time))
    else:
        print('test!!!!!!!!!!!!!!!!')
        # state = env.reset()
        state = env.reset()
        state = np.array(state)
        state = state.astype(np.float32)
        for i, agent in enumerate(agents):
            agent.load(args.load_dir, env.ddr_name[0])
            agent.policy_net([state[i]])

        count_wall = 0
        count_agent = 0
        for episode in range(50):
            state = env.reset()
            state = np.array(state)
            state = state.astype(np.float32)
            episode_reward = 0
            episode_safe = 0
            start = time.time()
            # 存储每个episode的数据
            reward_steps = []
            action_steps = []
            obs_steps = []
            safe_steps = []
            steps = 0
            time_start = time.strftime("%d-%H:%M:%S", time.localtime(time.time()))
            actions = [0, 0, 0]
            for step in range(400):
                for i, agent in enumerate(agents):
                    # action = agent.policy_net.get_action(state[i], EXPLORE_NOISE_SCALE, greedy=True)
                    action = agent.policy_net.get_action(actions, state[i], explore_noise_scale=0,
                                                         greedy=True)  # 2021-11-28-11:23
                    actions[i] = action
                bad_index = 2
                dist = [np.linalg.norm(state[i][0:2] - state[bad_index][0:2])
                        for i in range(env.agent_number)]
                avoid_index = np.argsort(dist)[1]  # 返回第二小的索引

                # print('bad_index', bad_index)
                a_bad = simple_pursuit(state[bad_index], state[avoid_index])
                a_bad = np.array([a_bad])
                a_bad = a_bad.astype(np.float32)
                actions[bad_index] = a_bad
                state, reward, done_n, safe, info = env.step(actions)
                state = np.array(state)
                state = state.astype(np.float32)
                episode_reward += np.sum(reward)
                episode_safe += np.sum(safe)
                # print('observation', observation)
                # 每个episode、每一步的数据
                reward_steps.append(reward)
                action_steps.append(actions)
                obs_steps.append(state)
                safe_steps.append(safe)
                if env.agent_colliside == True:
                    count_agent += 1
                    print('count_agent', count_agent)
                if env.wall_collide == True:
                    count_wall += 1
                    print('count_wall', count_wall)
                # if any(done_n) or (step == max_steps - 1) or env.done_reach:
                if any(done_n) or (step == max_steps - 1):
                    # if step == max_steps - 1:
                    print("Printing reward to file")
                    reward_st = np.append(reward_st, reward_per_episode)
                    safe_st = np.append(safe_st, safe_per_episode)
                    print(time.time() - start)
                    state_each_episode.append(obs_steps)
                    # print('aaa', obs_steps)
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {}  | Episode Safe: {}  |Running Time: {}\n'.format(
                    episode + 1, args.episodes, episode_reward, episode_safe,
                    time.time() - t_state
                )
            )
            model_dir = './save/video_test/data'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            sio.savemat(model_dir + '/td3_cbf_test_{}_{}.mat'.format(time_start, time.strftime("%d-%H:%M:%S",
                                                                                               time.localtime())),
                        {'episode_state': state_each_episode, 'count_wall': count_wall, 'count_agent': count_agent})

        if not os.path.exists(args.testData_dir):
            os.makedirs(args.testData_dir)
        sio.savemat(args.testData_dir + '/td3_nocbf_nod_test_{}.mat'.format(time.strftime("%d-%H:%M:%S",
                                                                                          time.localtime())),
                    {'episode_state': state_each_episode, 'count_wall': count_wall, 'count_agent': count_agent})


if __name__ == '__main__':
    main()
