import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.shared = nn.Linear(state_size, 50)

        self.advantage1 = nn.Linear(50, 50)
        self.advantage2 = nn.Linear(50, action_size)

        self.value1 = nn.Linear(50, 50)
        self.value2 = nn.Linear(50, 1)

    def forward(self, state):
        x = F.relu(self.shared(state))

        adv = F.relu(self.advantage1(x))
        adv = F.relu(self.advantage2(adv))

        val = F.relu(self.value1(x))
        val = F.relu(self.value2(val))

        adv = adv - adv.max()
        return val + adv
