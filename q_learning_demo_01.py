# Kham pha moi truong gym
import gym

# create bien moi truong
env = gym.make("MountainCar-v0")
env.reset()

# Kham pha moi truong
# Lay state hien tai sau khoi tao
print(env.state)

# env.render()
# Lay so action xe co the thuc hien 
print(env.action_space.n)

# Lay x toi thieu, toi da va van toc toi thieu toi da
print(env.observation_space.high)
print(env.observation_space.low)

# render thu
while True: 
    action = 2 # thu luon di vao ben phai
    new_state, reward, done, _ = env.step(action=action)
    print(f"New State: {new_state} | Reward: {reward}")

    env.render()

# input()