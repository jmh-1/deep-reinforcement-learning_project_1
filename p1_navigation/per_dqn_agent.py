import numpy as np
import random
from collections import namedtuple, deque
import model
import sum_tree
from importlib import reload
import dqn_agent
# reload(dqn_agent)
# reload(sum_tree)
# reload(model)
QNetwork = model.QNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
from IPython.core.debugger import set_trace
import pdb
import timeit
import sys
import math


BUFFER_SIZE = int(2**14)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent(dqn_agent.Agent):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, gamma=GAMMA, tau=TAU, lr=LR, qNetwork=QNetwork):
        super(Agent, self).__init__(state_size, action_size, seed, gamma=GAMMA, tau=TAU, lr=LR, qNetwork=QNetwork)
        self.a = .5
        self.b = .5
        self.a_init = self.a
        self.b_init = self.b

        self.memory = PorportionalPiorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, indexes, probs) tuples 
            gamma (float): discount factor
        """
        # exponents for adjusting prioritization and bias annealing. They anneal to 0 and 1 respectively over 150000 steps (500 episodes)
        self.a = max(self.a - BATCH_SIZE*self.a_init/150000, 0)
        self.b = min(self.b - BATCH_SIZE*self.b_init/150000, 1)
        states, actions, rewards, next_states, dones, indexes, probs = experiences
        
        self.optimizer.zero_grad()
        preds = self.qnetwork_local(states).gather(1,actions)
        targets = self.computeTargetQ(rewards, next_states, dones)
        td_error = (targets - preds).detach()
        for i in range(td_error.shape[0]):
            self.memory.update_priority(indexes[i].item(), td_error[i].item(), self.a)
        weights = ((1 / len(self.memory)) * (1 / probs))**self.b
        weights /= torch.max(weights)
        loss = F.mse_loss(targets * weights, preds * weights)
        loss = loss 
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     


class PorportionalPiorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a (float): value used to determine how much priority is used
        """
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.memory = sum_tree.SumTree(buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
#         self.a = a
        self.count = 0
        # max priority should be higher than real priority so each element is 
        # likely to get picked at least once
        self.max_priority = 300
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.insert(self.max_priority, e)
        if self.count < self.buffer_size:
            self.count += 1

    def update_priority(self, idx, err, a):
        val = (.1 + abs(err))**a
        self.memory.update(idx, val)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = []
        indexes = []
        probs = []
        for i in range(self.batch_size):
            try:
                sampleVal = np.random.uniform(high=self.memory.total)
                idx = self.memory.find_val_idx(sampleVal)
                exp = self.memory.data[idx]
                experiences.append(exp)
                indexes.append(idx)
                probs.append(self.memory.get_val(idx) / self.memory.total)
#                 set_trace()
            except:
                e = sys.exc_info()[0]
                print(self.memory.total)
                print(e)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        indexes_d = torch.from_numpy(np.vstack(indexes).astype(np.uint8)).to(device)
        probs_d = torch.from_numpy(np.vstack(probs).astype(np.float)).float().to(device)
        return (states, actions, rewards, next_states, dones, indexes_d, probs_d)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.count;
