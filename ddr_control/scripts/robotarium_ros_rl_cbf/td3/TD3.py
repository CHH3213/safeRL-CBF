import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from td3.model import Actor,Critic
from td3.utils import to_tensor,to_numpy
# from cbf_qp_layer import CBFQPLayer
from cbf_qp_layer_bk import CBFQPLayer


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477




class TD3(object):
    def __init__(
            self,
            args,
            env,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        self.device = torch.device(f"cuda:{args.device_num}" if args.cuda else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action,args.hidden_size).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim,args.hidden_size).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.args = args
        self.total_it = 0

        # CBF layer
        self.env = env
        self.cbf_layer = CBFQPLayer(env, args, args.Vp,args.Ve, args.r, args.K1, args.K2)
        self.use_cbf = args.use_cbf

    def select_action(self, state,other_state):
        state = to_tensor(state, torch.FloatTensor, self.device)
        other_state = to_tensor(other_state, torch.FloatTensor, self.device)
        expand_dim = len(state.shape) == 1
        # print(np.shape(other_state))
        if expand_dim:
                state = state.unsqueeze(0)
                other_state = other_state.unsqueeze(0)
                
        # print(np.shape(other_state))
        # action = self.actor(state).cpu().data.numpy().flatten()
        action = self.actor(state)
        if self.use_cbf:
            safe_action = self.get_safe_action(state, other_state, action)

        return safe_action.detach().cpu().numpy()[0] if (expand_dim and self.use_cbf) else action.detach().cpu().numpy()

    def train(self, replay_buffer, batch_size):
        self.total_it += 1

        # Sample replay buffer
        state, other_state_batch,action, reward, next_state, not_done = replay_buffer.sample(batch_size=batch_size)
        state = torch.FloatTensor(state).to(self.device)
        other_state_batch = torch.FloatTensor(other_state_batch).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        not_done = torch.FloatTensor(not_done).to(self.device).unsqueeze(1)
        # print(np.shape(state),np.shape(other_state_batch),np.shape(action),np.shape(next_state),np.shape(reward),np.shape(not_done))
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            if self.use_cbf:  # 使用CBF 层
                # print(next_action)
                next_action = self.get_safe_action(next_state,other_state_batch, next_action)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor losse
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return critic_loss.item(), actor_loss.item()
    
    def get_state(self, obs):
        """Given the observation, this function does the pre-processing necessary and returns the state.

        Parameters
        ----------
        obs_batch : ndarray or torch.tensor
            Environment observation.

        Returns
        -------
        state_batch : ndarray or torch.tensor
            State of the system.

        """

        expand_dims = len(obs.shape) == 1
        is_tensor = torch.is_tensor(obs)

        if is_tensor:
            dtype = obs.dtype
            device = obs.device
            obs = to_numpy(obs)

        if expand_dims:
            obs = np.expand_dims(obs, 0)

        state_batch = np.zeros((obs.shape[0], 3))
        state_batch[:, 0] = obs[:, 0]
        state_batch[:, 1] = obs[:, 1]
        # theta = np.arctan2(obs[:, 3], obs[:, 2])
        # state_batch[:, 2] = theta
        state_batch[:, 2] = obs[:, 2]
       

        if expand_dims:
            state_batch = state_batch.squeeze(0)

        return to_tensor(state_batch, dtype, device) if is_tensor else state_batch


    def get_safe_action(self, obs_batch, other_state_batch, action_batch):
        """Given a nominal action, returns a minimally-altered safe action to take.

        Parameters
        ----------
        obs_batch : torch.tensor
        action_batch : torch.tensor
        Returns
        -------
        safe_action_batch : torch.tensor
            Safe actions to be taken (cbf_action + action).
        """

        state_batch = self.get_state(obs_batch)
        other_state_batch = other_state_batch if torch.is_tensor(other_state_batch) else to_tensor(other_state_batch,dtype=state_batch.dtype,device=state_batch.device)
        safe_action_batch = self.cbf_layer.get_safe_action(state_batch, other_state_batch, action_batch)
        # print("shapoe",state_batch.shape[0] )
        # print("state_batch",np.shape(state_batch))
        # print("other_state_batch",np.shape(other_state_batch))
        # print("action_batch",np.shape(action_batch))
        return safe_action_batch



    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "/critic.pt")
        torch.save(self.critic_optimizer.state_dict(), filename + "/critic_optimizer.pt")

        torch.save(self.actor.state_dict(), filename + "/actor.pt")
        torch.save(self.actor_optimizer.state_dict(), filename + "/actor_optimizer.pt")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "/critic.pt"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "/critic_optimizer.pt"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "/actor.pt"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "/actor_optimizer.pt"))
        self.actor_target = copy.deepcopy(self.actor)
