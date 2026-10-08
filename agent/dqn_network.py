import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Neural network used by the DDQN agent.

    Input:
        16 values representing the 4x4 board.

    Output:
        16 Q-values, one for each possible action.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Input layer
            nn.Linear(16, 128),
            nn.ReLU(),

            # Hidden layer
            nn.Linear(128, 128),
            nn.ReLU(),

            # Hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer
            # One Q-value for each of 16 cells
            nn.Linear(64, 16)
        )

    def forward(self, state):
        """
        Forward pass through the network.

        Parameters:
            state: Board state

        Returns:
            Q-values for all 16 actions.
        """

        return self.network(state)