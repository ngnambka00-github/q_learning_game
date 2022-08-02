import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore")

env = gym.make("MountainCar-v0")
env.reset()

# Chi khoang q-table
q_table_size = [20, 20]
q_table_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size

# Ham chuyen doi tu real_state ve q_state
def convert_state(real_state):
    q_state = (real_state - env.observation_space.low) // q_table_segment_size
    return tuple(q_state.astype(np.int64))

# print(convert_state(env.reset()))

c_learning_rate = 0.1
c_discount_value = 0.9
q_table = np.random.uniform(low=-2, high=0, size=(q_table_size + [env.action_space.n]))
done = False
current_state = convert_state(env.reset())

while not done:
    # Lay argmax Q value cua current_state
    action = np.argmax(q_table[current_state])

    # Hanh dont theo action da lay
    next_real_state, reward, done, _ = env.step(action=action)

    if done: 
        # Kiem tra xem vi tri x co lon hon la co hay khong
        if next_real_state[0] > env.goal_position:
            print("Da den co")
        else: 
            print("Fail")

    else: 
        # Convert ve q_state
        next_state = convert_state(next_real_state)

        # Update Q value cho (current_state, action)
        current_q_value = q_table[current_state + (action, )]

        new_q_value = (1 - c_learning_rate) * current_q_value + c_learning_rate  * (reward + c_discount_value * np.max(q_table[next_state]))

        current_state = next_state

print(q_table.shape)