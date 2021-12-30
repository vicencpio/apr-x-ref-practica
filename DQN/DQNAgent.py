from copy import deepcopy
from matplotlib import pyplot as plt
import torch
import numpy as np
import gym
import highway_env

env = gym.make("highway-v0")

class DQNAgent:
    def __init__(self, env, main_network, buffer, reward_threshold, epsilon=0.1, eps_decay=0.99, batch_size=32):
        self.env = env
        self.main_network = main_network
        self.target_network = deepcopy(main_network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = 100
        self.initialize()
    
    def initialize(self):
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = self.env.reset()
        self.car_position = []
        self.epsilon_list = []
        self.losses = []
        
    def take_step(self, eps, mode='train'):
        if mode == 'explore': 
            action = self.env.action_space.sample()
        else:
            action = self.main_network.get_action(self.state0, eps)
            self.step_count += 1
            
        new_state, reward, done, _ = self.env.step(action)
        #env.render()
        self.total_reward += reward
        self.buffer.append(self.state0, action, reward, done, new_state)
        self.state0 = new_state.copy()
        
        if done:
            self.state0 = env.reset()
        return done
    
    def train(self, gamma=0.99, max_episodes=50000,
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000):
        
        self.gamma = gamma

        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(self.epsilon, mode='explore')

        episode = 0
        training = True
        print("Training...")
        while training:
            self.state0 = self.env.reset()
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                gamedone = self.take_step(self.epsilon, mode='train')
               
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.main_network.state_dict())
                    self.sync_eps.append(episode)
                
                if gamedone:        
                    self.losses.append(np.mean(self.update_loss))
                    
                    episode += 1
                    self.training_rewards.append(self.total_reward)
                    self.update_loss = []   
                                                           
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)
                    
                    print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {}\t\t".format(
                        episode, mean_rewards, self.epsilon), end="")

                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
                    
                    self.epsilon_list.append(self.epsilon)
                    
                    #plt.imshow(env.render(mode="rgb_array"))
                    

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch] 
        rewards_vals = torch.FloatTensor(rewards)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1)
        dones_t = torch.ByteTensor(dones)
        
        qvals = torch.gather(self.main_network.get_qvals(states), 1, actions_vals)
        qvals_next = torch.max(self.target_network.get_qvals(next_states),
                               dim=-1)[0].detach()
        qvals_next[dones_t] = 0
        
        expected_qvals = self.gamma * qvals_next + rewards_vals
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        return loss
    
    def update(self):
        self.main_network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.main_network.optimizer.step()
        self.update_loss.append(loss.detach().numpy())