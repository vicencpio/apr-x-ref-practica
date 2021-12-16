import gym
import highway_env

# Exercici 1.1
# Exploració de l'entorn

# Importem i inicialitzem l'entorn
env = gym.make("highway-v0")
env.reset()

print("Dimensió de l'espai d'accions: {} ".format(env.action_space))
print("Dimensió de l'espai d'estats: {} ".format(env.observation_space.shape[0]))

# Representació d'una acció aleatòria