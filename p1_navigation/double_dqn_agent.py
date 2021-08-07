from importlib import reload
from IPython.core.debugger import set_trace
import dqn_agent
# reload(dqn_agent)


class Agent(dqn_agent.Agent):

    def computeTargetQ(self, rewards, next_states, dones):
        next_action_indices = self.qnetwork_local(next_states).detach().max(1)[1]
        target_network_values = self.qnetwork_target(next_states).detach()
        next_action_values = target_network_values[range(target_network_values.shape[0]),next_action_indices]
        return rewards + self.gamma*next_action_values.unsqueeze(1)*(1-dones)

