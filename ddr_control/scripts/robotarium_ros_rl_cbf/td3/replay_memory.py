import random
import numpy as np


class ReplayMemory:

    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state,other_s, action, reward, next_state,not_done):
        """
        other_s: 额外添加， 保存其他agent和障碍物的位置
        """

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state,other_s, action, reward, next_state,not_done)
        self.position = (self.position + 1) % self.capacity

    def batch_push(self, state_batch,other_s_batch, action_batch, reward_batch, next_state_batch,not_done_batch):

        for i in range(state_batch.shape[0]):  # TODO: Optimize This
            self.push(state_batch[i],other_s_batch[i], action_batch[i], reward_batch[i], next_state_batch[i],not_done_batch[i])  # Append transition to memory

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        state, other_s, action, reward, next_state,not_done = map(np.stack, zip(*batch))
        return state, other_s, action, reward, next_state,not_done

    def __len__(self):
        return len(self.buffer)
