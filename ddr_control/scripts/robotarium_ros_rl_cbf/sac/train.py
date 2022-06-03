import time
from sac.replay_memory import ReplayMemory
from sac.utils import prGreen, prYellow
import numpy as np
from controllers.simple_pv import *
import matplotlib.pyplot as plt
import scipy.io as sio



def train(agent, env, args, experiment=None, index=0):
    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0

    all_rewards = []
    all_actions = []
    all_states = []
    all_o_states = []
    N = env.agent_number
    for i_episode in range(args.max_episodes):
        episode_reward = 0
        steps_reward = []
        steps_action = []
        steps_state = []
        steps_o_state = []
        episode_steps = 0
        done = False
        states = env.reset(False)
        obs,other_s = env.get_obs(index)
        actions = np.zeros((N, 2))
        actions[:, 0] = 0.15  # 线速度恒定
        while not done:
            start = time.time()
            if episode_steps % 10 == 0:
                prYellow(
                    'Episode {} - step {} - eps_rew {} '.format(i_episode, episode_steps, episode_reward
                                                                ))

            # If using model-based RL then we only need to have enough data for the real portion of the replay buffer
            if len(memory) > args.batch_size:

                # Number of updates per step in environment
                for i in range(args.updates_per_step):

                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates
                                                                                                         )

                    if experiment:
                        experiment.log_metric('loss/critic_1', critic_1_loss, updates)
                        experiment.log_metric('loss/critic_2', critic_2_loss, step=updates)
                        experiment.log_metric('loss/policy', policy_loss, step=updates)
                        experiment.log_metric('loss/entropy_loss', ent_loss, step=updates)
                        experiment.log_metric('entropy_temperature/alpha', alpha, step=updates)
                    updates += 1

            action = agent.select_action(obs, other_s,warmup=args.start_steps > total_numsteps)  # Sample action from policy
            for idx in range(0, N):
                if idx == index:
                    continue
                # actions[idx, 1] = simple_pursuit(states[idx], states[index])
                actions[idx, 1] = np.random.uniform(-1, 1)
                # actions[idx, 1] = CBF([actions[idx, 1]], all_states_, idx)
            # print(args.use_cbf)

            safe_a = action
            # print(safe_a)
            actions[index] = safe_a

            next_states = env.step(actions)
            reward, done, info = env._reward_done(index)
            next_obs,other_s = env.get_obs(index)
            states = next_states

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward
            steps_reward.append(reward)
            steps_action.append(action)
            steps_state.append(states)

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1 if episode_steps == env.max_episode_steps else float(not done)

            memory.push(obs,other_s, action, reward, next_obs, mask, t=episode_steps * env.dt,
                        next_t=(episode_steps + 1) * env.dt)  # Append transition to memory

            obs = next_obs
            # if (time.time() - start) + 1e-5 < env.dt:
            #     time.sleep(env.dt - (time.time() - start))

        all_rewards.append(steps_reward)
        all_actions.append(steps_action)
        all_states.append(steps_state)
        all_o_states.append(steps_o_state)

        # [optional] save intermediate model
        if i_episode % int(args.save_frenquency) == 0:
            agent.save_model(args.output)
            # 保存数据
            sio.savemat(args.output + '/reward.mat', {'rewards': all_rewards,
                                'actions': all_actions,
                                'states': all_states
                                })
        if experiment:
            # Comet.ml logging
            experiment.log_metric('reward/train', episode_reward, step=i_episode)
        prGreen("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                    episode_steps,
                                                                                    round(episode_reward, 2)
                                                                                    ))

        # Evaluation
        if (i_episode + 1) % 50 == 0 and args.eval is True:
            avg_reward = 0.
            episodes = 1
            for _ in range(episodes):
                states = env.reset(True)
                obs, other_s = env.get_obs(index)

                episode_reward = 0
                done = False
                while not done:
                    start = time.time()
                    action = agent.select_action(obs,other_s,
                                                 evaluate=True)  # Sample action from policy

                    all_states_ = np.vstack((states, env.hazards_locations))
                    for idx in range(0, N):
                        if idx == index:
                            continue
                        actions[idx, 1] = simple_pursuit(states[idx], states[index])
                        actions[idx, 0] = 0.15 * 0.95
                        # actions[idx, 1] = CBF([actions[idx, 1]], all_states_, idx)

                    safe_a = action
                    actions[index, 1] = safe_a
                    next_states = env.step(actions)
                    reward, done, info = env._reward_done(index)
                    next_obs,other_s = env.get_obs(index)
                    states = next_states
                    episode_reward += reward
                    obs = next_obs
                    if (time.time() - start) + 1e-5 < env.dt:
                        time.sleep(np.abs(env.dt - (time.time() - start)))

                avg_reward += episode_reward
                plt.close()
            avg_reward /= episodes
            if experiment:
                experiment.log_metric('avg_reward/test', avg_reward, step=i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)
                                                              ))
            print("----------------------------------------")