import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)

from experienceReplayBuffer import experienceReplayBuffer
from NeuralNet import NeuralNet
from DQNAgent import DQNAgent
import gym
import highway_env

# Importem i inicialitzem l'entorn
env = gym.make("highway-v0")

# Declarem els hiperparàmetres
lr = 0.001            #Velocitat d'aprenentatge
MEMORY_SIZE = 10000   #Màxima capacitat del buffer
MAX_EPISODES = 1000   #Nombre màxim d'episodis (l'agent ha d'aprendre abans d'arribar a aquest valor)
EPSILON = 1           #Valor inicial d'epsilon
EPSILON_DECAY = .99   #Decaïment d'epsilon
GAMMA = 1             #Valor gamma de l'equació de Bellman
BATCH_SIZE = 32       #Conjunt a agafar del buffer per a la xarxa neuronal
BURN_IN = 50          #Nombre d'episodis inicials usats per emplenar el buffer abans d'entrenar
DNN_UPD = 3           #Freqüència d'actualització de la xarxa neuronal 
DNN_SYNC = 1000       #Freqüència de sincronització de pesos entre la xarxa neuronal i la xarxa objectiu


# Carreguem la xarxa neuronal i entrenem l'agent
buffer = experienceReplayBuffer(memory_size=MEMORY_SIZE, burn_in=BURN_IN)
dqn = NeuralNet(env, learning_rate=lr)
DQNagent = DQNAgent(env, dqn, buffer, EPSILON, EPSILON_DECAY, BATCH_SIZE)
DQNagent.train(gamma=GAMMA, max_episodes=MAX_EPISODES, batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD, dnn_sync_frequency=DNN_SYNC)