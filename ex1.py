import gym
import highway_env
from matplotlib import pyplot as plt

# Exercici 1.1

# Importem i inicialitzem l'entorn
env = gym.make("highway-v0")
env.reset()

# Exploració de l'entorn
print("Dimensió de l'espai d'accions: {} ".format(env.action_space))
print("Dimensió de l'espai d'estats: {} ".format(env.observation_space.shape[0]))
print("Número de carrils: {} ".format(env.config["lanes_count"]))
print("Temps d'observació màxim: {} ".format(env.config["duration"]))
print("Recompensa de col·lisió: {} ".format(env.config["collision_reward"]))
print("Recompensa de carril de la dreta: {} ".format(env.config["right_lane_reward"]))
print("Recompensa de màxima velocitat: {} ".format(env.config["high_speed_reward"]))
print("Rang màxima velocitat: {} ".format(env.config["reward_speed_range"]))

# Representació d'una acció aleatòria
t, total_reward, done = 0, 0, False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print("Observation: {}, Action: {}".format(obs, action))
    total_reward += round(reward, 2)
    t += 1

plt.imshow(env.render(mode="rgb_array"))
print("Episode finished after {} timesteps and reward was {} ".format(t, round(total_reward, 2)))
env.close()