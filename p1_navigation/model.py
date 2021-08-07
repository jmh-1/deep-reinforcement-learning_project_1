import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython.core.debugger import set_trace

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
#         set_trace()
#         print(type(self))
#         print(isinstance(self, QNetwork))
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_size, 50)
        self.linear2 = nn.Linear(50, 50)
        self.linear3 = nn.Linear(50, action_size)
#         self.seq = nn.Sequential(
#             nn.Linear(state_size, 50),
#             nn.ReLU(inplace=True),
#             nn.Linear(50, 50),
#             nn.ReLU(inplace=True),
#             nn.Linear(50, action_size),
#         )
        "*** YOUR CODE HERE ***"

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.linear1(state)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        if math.isnan(x[0][0].item()):
            set_trace()
        return x
