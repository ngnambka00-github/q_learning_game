import gym
import numpy as np

# Loading paramter
data = np.load("./model/best_model_paramter.npz")
max_ep_reward = data['arr_0'][0]
max_start_state = data['arr_1'][0]
max_ep_action_list = data['arr_2']

env = gym.make("MountainCar-v0")

while True: 
    env.reset()
    env.state = max_start_state
    for action in max_ep_action_list:
        env.step(action)
        env.render()

    done = False
    while not done:
        _, _, done,_ = env.step(0)
        env.render()