import torch
import numpy as np
import gym
import highway_env
from matplotlib import pyplot as plt

env = gym.make("highway-v0")

class reinforceAgent:

    def __init__(self, env, main_network):
        self.env = env
        self.main_network = main_network
        self.nblock = 100
        self.reward_threshold = self.env.spec.reward_threshold
        
        self.initialize()

        
    def initialize(self):
        self.batch_rewards = []
        self.batch_actions = []
        self.batch_states = []
        self.batch_counter = 1
        self.training_rewards = []
        self.mean_training_rewards = []
        self.update_loss = []
        self.losses = []

        
    ## Entrenament
    def train(self, gamma=0.99, batch_size=10):
        self.gamma = gamma
        self.batch_size = batch_size
        
        episode = 0
        action_space = np.arange(self.env.action_space.n)
        training = True
        print("Training...")
        while training:
            state0 = env.reset()
            episode_states = [] 
            episode_rewards = [] 
            episode_actions = [] 
            gamedone = False
            
            while gamedone == False:
                action_probs = self.main_network.get_action_prob(state0).detach().numpy() # Distribució de probabilitat de les accions donat l'estat actual
                action = np.random.choice(action_space, p=action_probs) # Acció aleatòria de la distribució de probabilitat
                next_state, reward, gamedone, _ = env.step(action)
                
                episode_states.append(state0)
                episode_rewards.append(reward)
                episode_actions.append(action)
                state0 = next_state
                
                if gamedone:
                    episode += 1
                    # Calculem el terme del retorn menys la línia de base
                    self.batch_rewards.extend(self.discount_rewards(episode_rewards))
                    self.batch_states.extend(episode_states)
                    self.batch_actions.extend(episode_actions)
                    
                    self.training_rewards.append(sum(episode_rewards))
                    
                    if self.batch_counter == self.batch_size:
                        self.update(self.batch_states, self.batch_rewards, self.batch_actions)
                        
                        self.losses.append(np.mean(self.update_loss))
                        
                        self.update_loss = []
                    
                        self.batch_rewards = []
                        self.batch_actions = []
                        self.batch_states = []
                        self.batch_counter = 1
                    
                    self.batch_counter += 1
                    
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)
                    
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t".format(
                        episode, mean_rewards), end="")

                    plt.imshow(env.render(mode="rgb_array"))

                    
    def discount_rewards(self, rewards):
        discount_r = np.zeros_like(rewards)
        timesteps = range(len(rewards))
        reward_sum = 0
        for i in reversed(timesteps):  #revertim la direcció del vector per fer la suma cumulativa
            reward_sum = rewards[i] + self.gamma*reward_sum
            discount_r[i] = reward_sum
        baseline = np.mean(discount_r)/np.std(discount_r) # establim la mitjana de la recompensa estandaritzada com a línia de base
        return discount_r - baseline 
        
    
    def calculate_loss(self, state_t, action_t, reward_t):
        logprob = torch.log(self.main_network.get_action_prob(state_t))
        selected_logprobs = reward_t * \
                        logprob[np.arange(len(action_t)), action_t]
        loss = -selected_logprobs.mean()
        return loss
    
    def update(self, batch_s, batch_r, batch_a):
        self.main_network.optimizer.zero_grad()  #eliminem qualsevol gradient passat
        state_t = torch.FloatTensor(batch_s)
        reward_t = torch.FloatTensor(batch_r)       
        action_t = torch.LongTensor(batch_a)             
        loss = self.calculate_loss(state_t, action_t, reward_t) # calculem la pèrdua
        loss.backward() # fem la diferència per obtenir els gradients
        self.main_network.optimizer.step() # apliquem els gradients a la xarxa neuronal
        self.update_loss.append(loss.detach().numpy())   