import torch
import numpy as np

class NeuralNetReinforce(torch.nn.Module):

    def __init__(self, env, learning_rate=1e-3):
           
        super(NeuralNetReinforce, self).__init__()
        
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = learning_rate
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_inputs, 64, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 32, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 16, bias=True),
            torch.nn.Tanh(),
            torch.nn.Linear(16, self.n_outputs, bias=True),
            torch.nn.Softmax(dim=-1))
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def get_action_prob(self, state):
        action_probs = self.model(torch.FloatTensor(state))
        return action_probs