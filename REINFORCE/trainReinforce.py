import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)

from experienceReplayBuffer import experienceReplayBuffer
from NeuralNetReinforce import NeuralNetReinforce
from reinforceAgent import reinforceAgent
import gym
import highway_env

# configuració de l'observació
config = {
    "observation": {
    "type": "OccupancyGrid",
    "vehicles_count": 5,
    "features": ["presence", "x", "y", "vx", "vy"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20]
    },
    "grid_size": [[-10, 10], [-10, 10]],
    "grid_step": [5, 5],
    "absolute": False
    }
}

# Importem i inicialitzem l'entorn
env = gym.make("highway-v0")
env.configure(config)
env.reset()

# Definició hiperparàmetres
lr = 0.01
GAMMA = 0.99
BATCH_SIZE = 8

# Carregar model i entrenar agent
pgR = NeuralNetReinforce(env, learning_rate=lr)
r_agent = reinforceAgent(env, pgR)
r_agent.train(gamma=GAMMA, batch_size=BATCH_SIZE)