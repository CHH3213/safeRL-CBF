"""
Twin Delayed DDPG (TD3)-control barrier function
------------------------
"""

import argparse
import os
import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.layers.convolution import separable_conv
from tensorlayer.models import Model
from CBF_qpLayer import CBF

Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(
    description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train',
                    action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################
# choose env
ENV_ID = 'Pendulum-v0'  # environment id
RANDOM_SEED = 2  # random seed
RENDER = False  # render while training

# RL training
ALG_NAME = 'TD3'
TRAIN_EPISODES = 100  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 200  # maximum number of steps for one episode
BATCH_SIZE = 64  # update batch size
EXPLORE_STEPS = 500  # 500 for random action sampling in the beginning of training

HIDDEN_DIM = 64  # size of hidden layers for networks
UPDATE_ITR = 3  # repeated updates for single step
Q_LR = 3e-4  # q_net learning rate
POLICY_LR = 3e-4  # policy_net learning rate
# delayed steps for updating the policy network and target networks
POLICY_TARGET_UPDATE_INTERVAL = 3
EXPLORE_NOISE_SCALE = 1.0  # range of action noise for exploration
EVAL_NOISE_SCALE = 0.5  # range of action noise for evaluation of action value
REWARD_SCALE = 1.  # value range of reward
REPLAY_BUFFER_SIZE = 5e5  # size of replay buffer


###############################  TD3  ####################################


class ReplayBuffer:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, safe):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, action, reward, next_state, done, safe)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, safe = map(
            np.stack, zip(*batch))  # stack for each element
        """ 
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return state, action, reward, next_state, done, safe

    def __len__(self):
        return len(self.buffer)


class QNetwork(Model):
    """ the network for evaluate values of state-action pairs: Q(s,a) """

    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3):
        super(QNetwork, self).__init__()
        input_dim = num_inputs + num_actions
        # w_init = tf.keras.initializers.glorot_normal(seed=None)
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu,
                             W_init=w_init, in_channels=input_dim, name='q1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu,
                             W_init=w_init, in_channels=hidden_dim, name='q2')
        self.linear3 = Dense(n_units=1, W_init=w_init,
                             in_channels=hidden_dim, name='q3')

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(Model):
    """ the network for generating non-deterministic (Gaussian distributed) action from the state input """

    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        w_init = tf.random_uniform_initializer(-init_w, init_w)

        self.linear1 = Dense(n_units=hidden_dim, act=tf.nn.relu,
                             W_init=w_init, in_channels=num_inputs, name='policy1')
        self.linear2 = Dense(n_units=hidden_dim, act=tf.nn.relu,
                             W_init=w_init, in_channels=hidden_dim, name='policy2')
        self.linear3 = Dense(n_units=hidden_dim, act=tf.nn.relu,
                             W_init=w_init, in_channels=hidden_dim, name='policy3')
        self.output_linear = Dense(
            n_units=num_actions, W_init=w_init, b_init=tf.random_uniform_initializer(
                -init_w, init_w),
            in_channels=hidden_dim, name='policy_output'
        )
        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        x = self.linear1(state)
        x = self.linear2(x)
        x = self.linear3(x)
        output = tf.nn.tanh(self.output_linear(x))  # unit range output [-1, 1]
        return output

    def evaluate(self, actions, state_batch, eval_noise_scale):
        """ 
        generate action with state for calculating gradients;
        eval_noise_scale: as the trick of target policy smoothing, for generating noisy actions.
        action ,state 是一个batch的
        """
        state_batch = state_batch.astype(np.float32)
        action_batch = self.forward(state_batch)

        action_batch = self.action_range * action_batch

        # add noise
        normal = Normal(0, 1)
        eval_noise_clip = 2 * eval_noise_scale
        noise = normal.sample(action_batch.shape) * eval_noise_scale
        noise = tf.clip_by_value(noise, -eval_noise_clip, eval_noise_clip)
        action_batch = action_batch + noise
        # print(np.shape(action_batch))  # (64, 1)
        # print(type(action_batch))  # <class 'tensorflow.python.framework.ops.EagerTensor'>

        # TODO:qp layer
        # 非常耗时
        # print(state[5:7] + state[0:2])
        # print([state[18]])
        # print(state[0:3])
        # action_cbf = []
        for i, state in enumerate(state_batch):
            action = action_batch[i]
            qp_state = np.concatenate((state[5:7] + state[0:2], [state[18]], state[0:3], actions[2],
                                       state[3:5], [state[17]], actions[1], state[13:15], state[15:17]))
            action = CBF(action, qp_state)
            action = np.reshape(action, (1, 1))
            action = tf.convert_to_tensor(action, dtype=tf.float32)
            action_batch = tf.concat([action_batch[:i], action, action_batch[i + 1:]], axis=0)

        return action_batch

    def get_action(self, actions, state, explore_noise_scale, greedy=False):
        """ generate action with state for interaction with envronment """

        # print(state)
        # print(state[5:7] + state[0:2])
        action = self.forward([state])
        action = self.action_range * action.numpy()[0]
        # print(actions)
        # print(actions[2])
        # print(actions[1])
        if greedy:
            qp_state = np.concatenate((state[5:7] + state[0:2], [state[18]], state[0:3], actions[2],
                                       state[3:5], [state[17]], actions[1], state[13:15], state[15:17]))
            action = CBF(action.numpy(), qp_state)
            return action
        # add noise
        normal = Normal(0, 1)
        noise = normal.sample(action.shape) * explore_noise_scale
        action += noise
        # TODO:qp layer
        qp_state = np.concatenate((state[5:7] + state[0:2], [state[18]], state[0:3], actions[2],
                                   state[3:5], [state[17]], actions[1], state[13:15], state[15:17]))
        action = CBF(action.numpy(), qp_state)
        return action

    def sample_action(self):
        """ generate random actions for exploration """
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()


class TD3:

    def __init__(
            self, state_dim, action_dim, action_range, hidden_dim, replay_buffer, policy_target_update_interval=2,
            q_lr=3e-4, policy_lr=3e-4
    ):
        self.replay_buffer = replay_buffer

        # initialize all networks
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(
            state_dim, action_dim, hidden_dim, action_range)
        self.target_policy_net = PolicyNetwork(
            state_dim, action_dim, hidden_dim, action_range)
        # TODO: SAFE_VALUE has only two values, rewards have three values
        print('Q Network (1,2): ', self.q_net1)
        print('Policy Network: ', self.policy_net)

        # initialize weights of target networks
        self.target_q_net1 = self.target_ini(self.q_net1, self.target_q_net1)
        self.target_q_net2 = self.target_ini(self.q_net2, self.target_q_net2)
        self.target_policy_net = self.target_ini(
            self.policy_net, self.target_policy_net)

        # set train mode
        self.q_net1.train()
        self.q_net2.train()
        self.target_q_net1.eval()
        self.target_q_net2.eval()
        self.policy_net.train()
        self.target_policy_net.eval()

        self.update_cnt = 0
        self.policy_target_update_interval = policy_target_update_interval

        self.q_optimizer1 = tf.optimizers.Adam(q_lr)
        self.q_optimizer2 = tf.optimizers.Adam(q_lr)

        self.policy_optimizer = tf.optimizers.Adam(policy_lr)

    def target_ini(self, net, target_net):
        """ hard-copy update for initializing target networks """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(param)
        return target_net

    def target_soft_update(self, net, target_net, soft_tau):
        """ soft update the target net with Polyak averaging """
        for target_param, param in zip(target_net.trainable_weights, net.trainable_weights):
            target_param.assign(  # copy weight value into target parameters
                target_param * (1.0 - soft_tau) + param * soft_tau
            )
        return target_net

    def update(self, actions, batch_size, eval_noise_scale, reward_scale=10., gamma=0.9, soft_tau=1e-2, safe_scale=10.):
        """ update all networks in TD3 """
        self.update_cnt += 1
        state, action, reward, next_state, done, safe = self.replay_buffer.sample(
            batch_size)

        reward = reward[:, np.newaxis]  # expand dim
        done = done[:, np.newaxis]
        # safe = safe[:, np.newaxis]
        # print(np.shape(action))
        new_next_action = self.target_policy_net.evaluate(actions,
                                                          next_state, eval_noise_scale=eval_noise_scale
                                                          )  # clipped normal noise
        # print(np.shape(new_next_action))
        reward = reward_scale * (reward - np.mean(reward, axis=0)) / (
                np.std(reward, axis=0) + 1e-6
        )  # normalize with batch mean and std; plus a small number to prevent numerical problem
        # safe = safe_scale * (safe - np.mean(safe, axis=0)) / (
        #         np.std(safe, axis=0) + 1e-6
        # )
        # print(safe)
        # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Training Q Function
        # the dim 0 is number of samples
        target_q_input = tf.concat([next_state, new_next_action], 1)
        target_q_min = tf.minimum(self.target_q_net1(
            target_q_input), self.target_q_net2(target_q_input))
        # if done==1, only reward
        target_q_value = reward + (1 - done) * gamma * target_q_min
        q_input = tf.concat([state, action], 1)  # input of q_net

        with tf.GradientTape() as q1_tape:
            predicted_q_value1 = self.q_net1(q_input)
            q_value_loss1 = tf.reduce_mean(
                tf.square(predicted_q_value1 - target_q_value))
        q1_grad = q1_tape.gradient(
            q_value_loss1, self.q_net1.trainable_weights)
        self.q_optimizer1.apply_gradients(
            zip(q1_grad, self.q_net1.trainable_weights))

        with tf.GradientTape() as q2_tape:
            predicted_q_value2 = self.q_net2(q_input)
            q_value_loss2 = tf.reduce_mean(
                tf.square(predicted_q_value2 - target_q_value))
        q2_grad = q2_tape.gradient(
            q_value_loss2, self.q_net2.trainable_weights)
        self.q_optimizer2.apply_gradients(
            zip(q2_grad, self.q_net2.trainable_weights))

        # safe = tf.convert_to_tensor(safe, dtype=tf.float32)
        # print(safe)
        # Training Policy Function
        if self.update_cnt % self.policy_target_update_interval == 0:
            with tf.GradientTape() as p_tape:
                new_action = self.policy_net.evaluate(actions,
                                                      state, eval_noise_scale=0.0
                                                      )  # no noise, deterministic policy gradients
                new_q_input = tf.concat([state, new_action], 1)

                # """ implementation 1 """
                predicted_new_q_value = tf.minimum(self.q_net1(new_q_input), self.q_net2(new_q_input))
                # predicted_new_safe_value = tf.minimum(self.safe_net1(new_safe_input), self.safe_net2(new_safe_input))
                # """ implementation 2 """
                # predicted_new_q_value = self.q_net1(new_q_input)

                # TODO: safe critic and critic problems
                # policy_loss = -tf.reduce_mean(0.8*predicted_new_q_value + 0.2*predicted_new_safe_value)
                # print(predicted_new_q_value)
                policy_loss = - tf.reduce_mean(predicted_new_q_value)

            p_grad = p_tape.gradient(
                policy_loss, self.policy_net.trainable_weights)
            self.policy_optimizer.apply_gradients(
                zip(p_grad, self.policy_net.trainable_weights))

            # Soft update the target nets
            self.target_q_net1 = self.target_soft_update(
                self.q_net1, self.target_q_net1, soft_tau)
            self.target_q_net2 = self.target_soft_update(
                self.q_net2, self.target_q_net2, soft_tau)
            self.target_policy_net = self.target_soft_update(
                self.policy_net, self.target_policy_net, soft_tau)

    def save(self, path, AGENT):  # save trained weights
        # path = os.path.join('model', '_'.join([ALG_NAME, AGENT]))
        if not os.path.exists(path):
            os.makedirs(path)

        def extend_path(s): return os.path.join(path, s)

        tl.files.save_npz(self.q_net1.trainable_weights,
                          extend_path('{}_model_q_net1.npz'.format(AGENT)))
        tl.files.save_npz(self.q_net2.trainable_weights,
                          extend_path('{}_model_q_net2.npz'.format(AGENT)))
        tl.files.save_npz(self.target_q_net1.trainable_weights, extend_path(
            '{}_model_target_q_net1.npz'.format(AGENT)))
        tl.files.save_npz(self.target_q_net2.trainable_weights, extend_path(
            '{}_model_target_q_net2.npz'.format(AGENT)))
        tl.files.save_npz(self.policy_net.trainable_weights,
                          extend_path('{}_model_policy_net.npz'.format(AGENT)))
        tl.files.save_npz(self.target_policy_net.trainable_weights,
                          extend_path('{}_model_target_policy_net.npz'.format(AGENT)))

    def load(self, path, AGENT):  # load trained weights
        # path = os.path.join('model', '_'.join([ALG_NAME, AGENT]))
        def extend_path(s): return os.path.join(path, s)

        tl.files.load_and_assign_npz(extend_path(
            '{}_model_q_net1.npz'.format(AGENT)), self.q_net1)
        tl.files.load_and_assign_npz(extend_path(
            '{}_model_q_net2.npz'.format(AGENT)), self.q_net2)
        tl.files.load_and_assign_npz(extend_path(
            '{}_model_target_q_net1.npz'.format(AGENT)), self.target_q_net1)
        tl.files.load_and_assign_npz(extend_path(
            '{}_model_target_q_net2.npz'.format(AGENT)), self.target_q_net2)
        tl.files.load_and_assign_npz(extend_path(
            '{}_model_policy_net.npz'.format(AGENT)), self.policy_net)
        tl.files.load_and_assign_npz(extend_path('{}_model_target_policy_net.npz'.format(AGENT)),
                                     self.target_policy_net)


if __name__ == '__main__':
    # initialization of env
    env = gym.make(ENV_ID).unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # scale action, [-action_range, action_range]
    action_range = env.action_space.high

    # reproducible
    env.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # initialization of buffer
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    # initialization of trainer
    agent = TD3(
        state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer, POLICY_TARGET_UPDATE_INTERVAL, Q_LR, POLICY_LR
    )
    t0 = time.time()

    # training loop
    if args.train:
        frame_idx = 0
        all_episode_reward = []

        # need an extra call here to make inside functions be able to use model.forward
        state = env.reset().astype(np.float32)
        agent.policy_net([state])
        agent.target_policy_net([state])

        for episode in range(TRAIN_EPISODES):
            state = env.reset().astype(np.float32)
            episode_reward = 0

            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                if frame_idx > EXPLORE_STEPS:
                    action = agent.policy_net.get_action(
                        state, EXPLORE_NOISE_SCALE)
                else:
                    action = agent.policy_net.sample_action()

                next_state, reward, done, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                done = 1 if done is True else 0

                replay_buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                frame_idx += 1

                if len(replay_buffer) > BATCH_SIZE:
                    for i in range(UPDATE_ITR):
                        agent.update(
                            BATCH_SIZE, EVAL_NOISE_SCALE, REWARD_SCALE)

                if done:
                    break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(
                    all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
        agent.save()
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        agent.load()

        # need an extra call here to make inside functions be able to use model.forward
        state = env.reset().astype(np.float32)
        agent.policy_net([state])

        for episode in range(TEST_EPISODES):
            state = env.reset().astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                action = agent.policy_net.get_action(
                    state, EXPLORE_NOISE_SCALE, greedy=True)
                state, reward, done, info = env.step(action)
                state = state.astype(np.float32)
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0
                )
            )
