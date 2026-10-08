import random
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for DDQN.

    Stores:
        state
        action
        reward
        next_state
        done
    """

    def __init__(self, capacity=100000):
        """
        Parameters:
            capacity: Maximum number of experiences to store.
        """

        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Store one experience.
        """

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        """

        batch = random.sample(
            self.buffer,
            batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        """Return number of stored experiences."""

        return len(self.buffer)