import torch
import numpy as np

class NeuralNet(torch.nn.Module):  

    def __init__(self, env, learning_rate=1e-3):
        super(NeuralNet, self).__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(25, 16, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.n_outputs, bias=True))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
          
    ### e-greedy method
    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)  # acció random
        else:
            qvals = self.get_qvals(state)  # acció del càlcul de Q per a aquesta acció
            action= torch.max(qvals, dim=-1)[1].item()
        return action
    
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state)
        return self.model(state_t)