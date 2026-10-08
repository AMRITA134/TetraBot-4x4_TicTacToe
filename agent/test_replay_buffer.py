import numpy as np

from replay_buffer import ReplayBuffer


buffer = ReplayBuffer(capacity=100)


state = np.zeros(16)
next_state = np.ones(16)

buffer.push(
    state,
    5,
    0,
    next_state,
    False
)

print("Buffer size:", len(buffer))


states, actions, rewards, next_states, dones = buffer.sample(1)

print("State:", states[0])
print("Action:", actions[0])
print("Reward:", rewards[0])
print("Next state:", next_states[0])
print("Done:", dones[0])